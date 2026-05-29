"""Pool.initialize must not commit connections into a closed pool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_close_race_routes_survivors_through_cleanup() -> None:
    """close() landing while gather is suspended must close every survivor."""
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)
    gather_started = asyncio.Event()
    close_done = asyncio.Event()

    async def _create() -> object:
        gather_started.set()
        await close_done.wait()
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    init_task = asyncio.create_task(pool.initialize())
    await gather_started.wait()
    # Extra yields so all three gather branches are parked.
    for _ in range(5):
        await asyncio.sleep(0)

    # Race close past the mid-gather initialize, then let gather resolve.
    await pool.close()
    close_done.set()

    await asyncio.wait_for(init_task, timeout=1.0)

    assert pool._pool.qsize() == 0, (
        f"leaked {pool._pool.qsize()} connections in closed pool's queue"
    )
    assert pool._size == 0, f"_size did not return to zero: {pool._size}"
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"connection {i} was never closed (leaked transport)"


@pytest.mark.asyncio
async def test_initialize_close_race_guards_initialized_flag() -> None:
    """If close() interrupted the put-loop, _initialized stays False."""
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)
    gather_started = asyncio.Event()
    close_done = asyncio.Event()

    async def _create() -> object:
        gather_started.set()
        await close_done.wait()
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    init_task = asyncio.create_task(pool.initialize())
    await gather_started.wait()
    for _ in range(5):
        await asyncio.sleep(0)

    await pool.close()
    close_done.set()

    await asyncio.wait_for(init_task, timeout=1.0)

    assert pool._initialized is False, (
        "_initialized became True despite close() interrupting the put-loop"
    )


@pytest.mark.asyncio
async def test_initialize_close_between_gather_and_phase_c_closes_all_survivors() -> None:
    """close() landing between gather return and Phase C's lock acquire
    must route every survivor through the shielded close helper."""
    pool = ConnectionPool(["n1:9001", "n2:9001", "n3:9001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)

    async def _create() -> object:
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    # Intercept Phase C's lock acquire to run close() before it returns.
    # Drop the intercept on first call so close()'s own acquire won't re-enter.
    original_acquire = pool._lock.acquire
    intercept_fired = asyncio.Event()
    seen_acquires = 0

    async def _intercepted_acquire() -> bool:
        nonlocal seen_acquires
        seen_acquires += 1
        # Phase A is acquire #1; Phase C is acquire #2 (all-success path).
        if seen_acquires == 2 and not intercept_fired.is_set():
            intercept_fired.set()
            # Restore before close() so its own acquire goes through normally.
            pool._lock.acquire = original_acquire
            await pool.close()
        return await original_acquire()

    pool._lock.acquire = _intercepted_acquire  # type: ignore[assignment]

    await asyncio.wait_for(pool.initialize(), timeout=1.0)
    assert intercept_fired.is_set(), "Phase C lock acquire was not intercepted"

    assert pool._initialized is False
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"survivor {i} was never closed"
    assert pool._closed is True
    assert pool._size == 0, f"_size did not return to zero: {pool._size}"
