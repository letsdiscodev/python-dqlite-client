"""``pool.acquire()`` must not yield a fresh connection from the fresh-slot
reservation arm on a closed pool.

A close() racing the in-flight ``_create_connection`` must, via the
post-create ``_closed`` re-check, raise ``DqliteConnectionError("Pool is
closed")`` and close the freshly-built connection (no transport leak).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_fresh_slot_raises_on_close_race() -> None:
    """close() landing between the fresh-slot _create_connection suspension
    and the yield must raise DqliteConnectionError and close the fresh conn."""
    pool = ConnectionPool(["localhost:19001"], min_size=0, max_size=1, timeout=2.0)

    # Empty queue, fresh-slot arm taken (size=0, max_size=1).
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

    # Wait for the fresh-slot arm to reach _create_connection.
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
    # Reservation slot must be released, else the pool wedges at max_size
    # with nothing checked out. The close-awaited assertion alone misses this.
    assert pool._size == 0, f"reservation slot leaked on close-race: pool._size={pool._size}"
