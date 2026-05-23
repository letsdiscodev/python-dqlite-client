"""Pin: ``DqliteConnection.close()`` on a pool-released connection
in a forked child runs the fork-cleanup branch (nulling
``_pending_drain`` and the rest of the parent-loop-bound state)
rather than taking the ``_pool_released`` short-circuit.

Background: the SA pool's checkin path leaves a conn marked
``_pool_released=True`` so a follow-up ``close()`` is a same-
process no-op. The fork branch (per ISSUE-944) nulls inherited
parent-loop state (``_pending_drain`` Task is the canonical leak
shape). Previously the ``_pool_released`` short-circuit ran
BEFORE the fork branch, pre-empting the cleanup — pool-released
conns inherited across fork kept their parent-loop ``_pending_drain``
references alive in the child, tripping
"Task was destroyed but it is pending" warnings at child GC and
retaining the inherited writer transport (and FD) indefinitely.

The fix reorders ``close()`` so the fork detection runs first,
regardless of ``_pool_released``. Same-process idempotency
(``_pool_released=True`` -> no-op) is preserved.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_pool_released_close_in_forked_child_nulls_pending_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set ``_pool_released=True`` and a sentinel
    ``_pending_drain`` Task, simulate fork by monkeypatching
    ``_current_pid``, then call ``close()``. The fork branch MUST
    null ``_pending_drain`` even though the connection is pool-
    released — pre-empting that with the short-circuit was the
    documented leak shape.
    """
    conn = DqliteConnection("leader:9001")
    conn._pool_released = True
    sentinel_task = asyncio.create_task(asyncio.sleep(60))
    conn._pending_drain = sentinel_task
    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()

    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    try:
        await conn.close()
    finally:
        sentinel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sentinel_task

    # The fork branch must null _pending_drain (matching the symmetric
    # fork branch nullification). The same set of fields is nulled.
    assert conn._pending_drain is None, (
        "DqliteConnection.close() in a forked child with _pool_released=True "
        "must null _pending_drain — the prior ordering pre-empted the fork "
        "branch and left the parent-loop Task referenced through child GC."
    )
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._bound_loop_ref is None
    assert conn._protocol is None
    assert conn._db_id is None
    assert conn._closed is True


@pytest.mark.asyncio
async def test_pool_released_close_same_process_is_still_no_op() -> None:
    """Regression guard: same-process ``close()`` on a pool-released
    conn must still no-op (the SA pool checkin idempotency contract).
    The reorder must NOT scrub state when same-process +
    ``_pool_released=True``.
    """
    conn = DqliteConnection("leader:9001")
    conn._pool_released = True
    # Plant a sentinel _pending_drain to detect any scrubbing.
    sentinel_task = asyncio.create_task(asyncio.sleep(60))
    conn._pending_drain = sentinel_task
    conn._in_transaction = True
    try:
        await conn.close()

        # Same-process + pool-released: nothing must be scrubbed.
        assert conn._pending_drain is sentinel_task
        assert conn._in_transaction is True
    finally:
        sentinel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sentinel_task


@pytest.mark.asyncio
async def test_non_pool_released_fork_branch_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: the fork branch must continue to fire for
    connections that are NOT pool-released (the original
    ISSUE-944 path)."""
    conn = DqliteConnection("leader:9001")
    sentinel_task = asyncio.create_task(asyncio.sleep(60))
    conn._pending_drain = sentinel_task
    conn._in_transaction = True

    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    try:
        await conn.close()
    finally:
        sentinel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sentinel_task

    assert conn._pending_drain is None
    assert conn._in_transaction is False
    assert conn._closed is True
