"""Cancellation during _reset_connection ROLLBACK must propagate
(CancelledError is outside _POOL_CLEANUP_EXCEPTIONS) and let
_release's finally keep slot accounting consistent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_reset_rollback_cancelled_invalidates_and_releases_slot() -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")

    started = asyncio.Event()
    suspend = asyncio.Event()

    async def slow_rollback(sql: str) -> object:
        started.set()
        try:
            await suspend.wait()
        except asyncio.CancelledError:
            raise
        return None

    with (
        patch.object(DqliteConnection, "is_connected", new=True),
        patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        patch.object(conn, "execute", new=slow_rollback),
    ):
        conn._in_transaction = True

        reset_task = asyncio.get_running_loop().create_task(pool._reset_connection(conn))
        await started.wait()
        reset_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reset_task

    assert pool._size in (0, 1)
    suspend.set()
