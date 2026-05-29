"""``ConnectionPool.acquire`` cleanup invariants under cancel churn: ``_size``
must not drift, and get_task / closed_task must not be orphaned."""

from __future__ import annotations

import asyncio
import gc
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_size_invariant_survives_cancel_churn() -> None:
    """Under N concurrent acquires each cancelled by an outer timeout, ``_size``
    must not drift."""
    pool = ConnectionPool(["localhost:9001"], max_size=2, min_size=0, timeout=5.0)

    mock_conn = MagicMock(spec=DqliteConnection)
    mock_conn.is_connected = True
    mock_conn.close = AsyncMock()
    mock_conn._in_transaction = False
    mock_conn._in_use = False
    mock_conn._bound_loop_ref = None
    mock_conn._pool_released = False
    mock_conn._address = "localhost:9001"
    mock_conn._check_in_use = MagicMock()

    with patch.object(pool._cluster, "connect", return_value=mock_conn):
        # Saturate the pool with one held conn.
        held_acquired = asyncio.Event()
        release_held = asyncio.Event()

        async def hold() -> None:
            async with pool.acquire():
                held_acquired.set()
                await release_held.wait()

        held_task = asyncio.create_task(hold())
        await held_acquired.wait()

        async def waiter() -> None:
            try:
                async with asyncio.timeout(0.05):
                    async with pool.acquire():
                        pass
            except (TimeoutError, asyncio.CancelledError):
                pass

        waiters = [asyncio.create_task(waiter()) for _ in range(20)]
        await asyncio.gather(*waiters, return_exceptions=True)

        release_held.set()
        await held_task

        # Must be exactly 0: a leaked reservation would pass the looser
        # 0 <= _size <= max_size bound.
        assert pool._size == 0
        assert pool._pool.qsize() == 0


@pytest.mark.asyncio
async def test_acquire_does_not_leak_pending_tasks(caplog: pytest.LogCaptureFixture) -> None:
    """get_task / closed_task must not be orphaned under cancel churn (orphans
    emit "Task exception was never retrieved" at GC)."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, min_size=0, timeout=5.0)

    mock_conn = MagicMock(spec=DqliteConnection)
    mock_conn.is_connected = True
    mock_conn.close = AsyncMock()
    mock_conn._in_transaction = False
    mock_conn._in_use = False
    mock_conn._bound_loop_ref = None
    mock_conn._pool_released = False
    mock_conn._address = "localhost:9001"
    mock_conn._check_in_use = MagicMock()

    with patch.object(pool._cluster, "connect", return_value=mock_conn):
        held_acquired = asyncio.Event()
        release_held = asyncio.Event()

        async def hold() -> None:
            async with pool.acquire():
                held_acquired.set()
                await release_held.wait()

        held_task = asyncio.create_task(hold())
        await held_acquired.wait()

        async def waiter() -> None:
            try:
                async with asyncio.timeout(0.05):
                    async with pool.acquire():
                        pass
            except (TimeoutError, asyncio.CancelledError):
                pass

        for _ in range(3):
            waiters = [asyncio.create_task(waiter()) for _ in range(5)]
            await asyncio.gather(*waiters, return_exceptions=True)

        release_held.set()
        await held_task

        # Flush any orphaned-task __del__ messages.
        gc.collect()
        await asyncio.sleep(0)

    leaked = [r for r in caplog.records if "Task exception was never retrieved" in r.getMessage()]
    assert not leaked, (
        f"acquire cleanup leaked {len(leaked)} unobserved task exceptions: {leaked!r}"
    )
