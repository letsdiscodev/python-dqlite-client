"""``close()`` on a pool-released connection in a forked child must run the fork-cleanup
branch (nulling inherited parent-loop state) rather than the ``_pool_released``
short-circuit; fork detection runs first while same-process idempotency is preserved.
Previously the short-circuit pre-empted cleanup, keeping the inherited ``_pending_drain``
Task and writer FD alive in the child (ISSUE-944)."""

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
    """The fork branch must null ``_pending_drain`` even when the connection is
    pool-released — pre-empting that with the short-circuit was the leak shape."""
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
    """Same-process ``close()`` on a pool-released conn must still no-op (SA pool checkin
    idempotency); the reorder must not scrub state."""
    conn = DqliteConnection("leader:9001")
    conn._pool_released = True
    sentinel_task = asyncio.create_task(asyncio.sleep(60))
    conn._pending_drain = sentinel_task
    conn._in_transaction = True
    try:
        await conn.close()

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
    """The fork branch must still fire for connections that are not pool-released."""
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
