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

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_publish_failure_releases_reservations_and_closes_survivors() -> None:
    """Phase C publish failure: simulate a synthetic failure inside
    ``put_nowait`` (e.g. a queue-invariant break). Every reserved
    slot must be released and every gathered survivor must be
    closed; the original failure propagates to the caller.

    The pre-fix shape had an await-driven put-loop where
    ``CancelledError`` could land between iterations and leave a
    partial queue + drifted ``_size``. The post-fix Phase C is
    atomic under ``self._lock`` with no awaits, so the
    cancel-mid-iteration surface is gone (by design). The
    equivalent semantic — "any abort path releases all
    reservations and closes all survivors" — is exercised here via
    the Phase C exception arm.
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

    # Patch ``put_nowait`` to raise on the third call: the first two
    # succeed but Phase C's loop raises and routes ALL successes
    # through the close helper (the publish-failure cleanup is
    # all-or-nothing, mirroring atomic-publish semantics).
    original_put_nowait = pool._pool.put_nowait
    put_call_count = 0

    def _put_nowait(conn: object) -> None:
        nonlocal put_call_count
        put_call_count += 1
        if put_call_count == 3:
            raise RuntimeError("synthetic Phase C failure")
        original_put_nowait(conn)  # type: ignore[arg-type]

    pool._pool.put_nowait = _put_nowait  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="synthetic Phase C failure"):
        await pool.initialize()

    # Atomic publish: on Phase C failure, ALL successes are routed
    # through the close helper; the queue ends up empty (the 2
    # partially-published conns were drained by the close-walk).
    # Reservation count returns to zero.
    assert pool._size == 0, f"_size should be 0, got {pool._size}"
    # Every gathered survivor was closed.
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"survivor {i} was not closed"
    assert pool._initialized is False


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
