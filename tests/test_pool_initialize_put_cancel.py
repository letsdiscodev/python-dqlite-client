"""Pool.initialize must keep _size accounting consistent if CancelledError
fires mid put-loop.

Before the fix, the sequence "gather succeeds → 2 of 5 puts complete →
CancelledError lands on the 3rd put" released the full ``_min_size``
reservation in the finally, even though the 2 queued connections are
still alive and counted in the queue. The pool size accounting drifted
below the queue depth, so a subsequent ``acquire()`` could create a
new connection on top of the 2 already queued without counting them —
briefly exceeding ``_max_size``.

The fix tracks which successes actually reached the queue via a
``unqueued`` counter that decrements per successful ``put``. On any
abort the finally releases only the uncommitted slots and closes the
unqueued survivors so their transports do not leak.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_cancel_mid_put_keeps_size_consistent() -> None:
    """Simulate CancelledError landing on the 3rd put when min_size=5.

    Before the fix: queue has 2 connections, _size was decremented by
    5 (the full reservation) so _size ended negative or 0 while
    qsize() == 2.

    After the fix: queue has 2 connections, _size reflects those 2
    committed slots plus any uncommitted reservations are released.
    """
    pool = ConnectionPool(["localhost:19001"], min_size=5, max_size=5, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(5):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)

    async def _create() -> object:
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    # Wrap put so the third call raises CancelledError. The first two
    # legitimately land in the queue.
    original_put = pool._pool.put
    put_call_count = 0

    async def _put(conn: object) -> None:
        nonlocal put_call_count
        put_call_count += 1
        if put_call_count == 3:
            raise asyncio.CancelledError()
        await original_put(conn)  # type: ignore[arg-type]

    pool._pool.put = _put  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        await pool.initialize()

    # The 2 committed connections stay in the queue.
    assert pool._pool.qsize() == 2
    # _size reflects exactly what's in the queue (2 committed slots).
    # Before the fix, _size would be 0 or negative here.
    assert pool._size == 2
    # The 3 unqueued survivors (including the one whose put raised)
    # had their transports closed so no orphans leak.
    unqueued_mocks = mocks[2:]
    for m in unqueued_mocks:
        m.close.assert_awaited_once()
    # The 2 committed mocks stay alive — still in the queue.
    for m in mocks[:2]:
        m.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_clean_success_leaves_size_at_min_size() -> None:
    """Regression guard: the new ``unqueued`` bookkeeping must not
    over-release on the happy path."""
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)

    async def _create() -> object:
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    await pool.initialize()

    assert pool._pool.qsize() == 3
    assert pool._size == 3
    for m in mocks:
        m.close.assert_not_awaited()
