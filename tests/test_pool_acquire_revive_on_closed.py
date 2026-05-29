"""``pool.acquire()``'s dead-conn revive branch must not yield a fresh
connection on a closed pool: a close() racing the in-flight
``_create_connection`` must hit the post-revive ``_closed`` check rather
than hand user code a conn on an invisibly-closed pool.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_revive_on_closed_pool_raises() -> None:
    """close() landing between revive's _create_connection and yield must
    raise DqliteConnectionError and close the fresh connection."""
    pool = ConnectionPool(["localhost:19001"], min_size=0, max_size=1, timeout=0.5)

    # Seed one stale (dead) connection.
    dead_conn = MagicMock()
    dead_conn.is_connected = False
    dead_conn.close = AsyncMock()
    dead_conn._address = "localhost:19001"
    await pool._pool.put(dead_conn)
    pool._size += 1

    fresh_conn = MagicMock()
    fresh_conn.is_connected = True
    fresh_conn.close = AsyncMock()
    fresh_conn._address = "localhost:19001"
    fresh_conn._pool_released = False

    create_gate = asyncio.Event()

    async def _create() -> object:
        # Release the fresh conn only after close() has set _closed=True.
        await create_gate.wait()
        return fresh_conn

    pool._create_connection = _create  # type: ignore[assignment]

    async def _consume() -> None:
        async with pool.acquire() as conn:
            pytest.fail(f"acquire yielded on a closed pool: {conn!r}")

    consume_task = asyncio.create_task(_consume())

    # Wait for revive to reach _create_connection.
    for _ in range(50):
        await asyncio.sleep(0)
        if create_gate._waiters:  # consumer is parked on create_gate.wait()
            break

    # Race close() past the suspended _create_connection, then resume it.
    await pool.close()
    create_gate.set()

    with pytest.raises(DqliteConnectionError, match="Pool is closed"):
        await asyncio.wait_for(consume_task, timeout=1.0)

    fresh_conn.close.assert_awaited()
