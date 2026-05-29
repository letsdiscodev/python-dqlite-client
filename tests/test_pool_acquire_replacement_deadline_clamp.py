"""When ``acquire()`` dequeues a dead conn and creates a REPLACEMENT, that
create is clamped to the remaining acquire deadline. Exercises the
dead-conn-replacement ``asyncio.timeout`` arm (the fresh-slot arm is pinned
separately) and the forensic detail in the timeout error message.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_replacement_create_clamped_by_pool_timeout() -> None:
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)
    # Seed a dead conn so acquire takes the replacement arm.
    dead = MagicMock()
    dead.is_connected = False
    dead.close = AsyncMock()
    dead._pool_released = True
    pool._pool.put_nowait(dead)
    pool._size = 1  # the seeded conn occupies one reservation slot

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)  # far past pool.timeout so the clamp shows
        raise AssertionError("clamp not applied to replacement create")

    loop = asyncio.get_running_loop()
    started = loop.time()

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        patch.object(pool, "_drain_idle", new=AsyncMock()),
        pytest.raises(DqliteConnectionError, match="Timed out creating"),
    ):
        async with pool.acquire():
            pytest.fail("should not reach the with-body")

    elapsed = loop.time() - started
    assert elapsed < 0.6, (
        f"acquire() blocked for {elapsed:.3f}s under pool.timeout=0.1s on "
        f"the replacement create path; the asyncio.timeout clamp at "
        f"pool.py:1515 did not fire"
    )


@pytest.mark.asyncio
async def test_acquire_replacement_create_timeout_message_carries_pool_state() -> None:
    """The replacement-create timeout error carries pool-id, max_size, idle,
    checked_out and timeout for pool-saturation forensics."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)
    dead = MagicMock()
    dead.is_connected = False
    dead.close = AsyncMock()
    dead._pool_released = True
    pool._pool.put_nowait(dead)
    pool._size = 1

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied")

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        patch.object(pool, "_drain_idle", new=AsyncMock()),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with pool.acquire():
            pytest.fail("should not reach")

    msg = str(exc_info.value)
    assert "Timed out creating a fresh connection from the pool" in msg
    assert "max_size=1" in msg
    assert "idle=" in msg
    assert "checked_out=" in msg
    assert "timeout=0.1s" in msg
