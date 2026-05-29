"""Pin: ``_release``'s finally awaits ``_pending_drain`` even when already done(),
so its exception is observed (Exception swallowed at DEBUG, BaseException
re-raised) instead of leaking to asyncio's task-finaliser logger.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_release_finally_awaits_done_pending_drain_with_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A drain task that completed with an Exception before the finally runs is
    awaited so the exception is observed (no GC warning); DEBUG marker fires."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1

    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._pool_released = False

    async def _fail() -> None:
        raise OSError("simulated drain failure")

    drain_task = asyncio.create_task(_fail())
    # Spin the loop so _release observes the task as already done().
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert drain_task.done(), "drain task must be done before _release runs"
    conn._pending_drain = drain_task

    pool._reset_connection = AsyncMock(return_value=False)  # close-and-drop
    conn.close = AsyncMock(return_value=None)

    with caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"):
        await pool._release(conn)

    assert any("suppressed pending-drain exception" in r.message for r in caplog.records), (
        "DEBUG log marker must fire so a future drain bug isn't invisible"
    )
    assert drain_task.done()
    assert drain_task.exception() is not None


@pytest.mark.asyncio
async def test_release_finally_absorbs_pending_drain_cancellation() -> None:
    """A CancelledError at the shield(pending) await must not tear down the rest
    of _release's cleanup; the cleanup tail (_pool_released = True) still runs."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1

    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._pool_released = False

    async def _slow_drain() -> None:
        await asyncio.sleep(60)

    drain_task = asyncio.create_task(_slow_drain())
    conn._pending_drain = drain_task

    pool._reset_connection = AsyncMock(return_value=False)
    conn.close = AsyncMock(return_value=None)

    # Cancel before _release: the awaiter-side raise lands in the suppress arm.
    drain_task.cancel()

    await pool._release(conn)

    assert conn._pool_released is True, (
        "CancelledError absorbed; release must finish the cleanup tail"
    )


@pytest.mark.asyncio
async def test_release_finally_done_pending_drain_with_baseexception_reraises() -> None:
    """A drain task that completed with a BaseException-class exception
    propagates out of _release rather than being swallowed."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1

    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._pool_released = False

    class _SentinelBase(BaseException):
        pass

    async def _raise_base() -> None:
        raise _SentinelBase("BaseException-class drain failure")

    drain_task = asyncio.create_task(_raise_base())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert drain_task.done()
    conn._pending_drain = drain_task

    pool._reset_connection = AsyncMock(return_value=False)
    conn.close = AsyncMock(return_value=None)

    with pytest.raises(_SentinelBase):
        await pool._release(conn)
