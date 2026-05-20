"""Pin: ``_release``'s finally awaits a ``_pending_drain`` task even
when it is already ``done()`` — the prior fix dropped the
``if not pending.done():`` short-circuit so a done task's exception
(``Exception`` swallowed at DEBUG, ``BaseException`` re-raised) is
observed instead of leaking to asyncio's task-finaliser logger.

Three sub-arms previously unpinned:

1. Done task with ``Exception`` → DEBUG-logged, swallowed.
2. Done task with ``BaseException`` → propagates out of ``_release``.
3. Already-done at finally entry — the common case under the
   ``await asyncio.shield(conn.close())`` shield discipline.
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
    """A pending-drain task that completed with a regular ``Exception``
    BEFORE ``_release``'s finally runs must be awaited so the
    exception is observed (no "Task exception was never retrieved" at
    GC). The DEBUG log marker fires."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1

    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._pool_released = False

    async def _fail() -> None:
        raise OSError("simulated drain failure")

    drain_task = asyncio.create_task(_fail())
    # Spin the loop so the task completes — `_release` will observe it
    # as already done().
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
    # The done task's exception must have been observed (no warning at GC).
    assert drain_task.done()
    assert drain_task.exception() is not None


@pytest.mark.asyncio
async def test_release_finally_done_pending_drain_with_baseexception_reraises() -> None:
    """A pending-drain task that completed with a
    ``BaseException``-class exception (``KeyboardInterrupt`` /
    ``SystemExit`` / project-internal sentinel) must propagate out
    of ``_release`` rather than being silently swallowed. This is
    the very behavior change that motivated dropping the
    ``if not pending.done():`` short-circuit."""
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
