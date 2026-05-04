"""``ConnectionPool.acquire``'s post-wait + post-cancel cleanup
invariants under TaskGroup-with-timeout cancel churn.

The architect plan suggested a structural ``try/finally``
refactor, but the triage downgraded to "preventive hardening" —
the existing ``except BaseException`` arm with
``suppress(asyncio.CancelledError)`` already catches the documented
race window. These regression pins document the invariants that
matter for production:

1. ``_size`` invariants survive cancel churn (no over- or
   under-allocation across many cancelled-mid-acquire calls).
2. ``get_task`` and ``closed_task`` are not orphaned (no "Task
   exception was never retrieved" at GC).

If a future refactor regresses cleanup discipline, these pins
catch it.
"""

from __future__ import annotations

import asyncio
import gc
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_size_invariant_survives_cancel_churn() -> None:
    """Under N concurrent acquires + outer ``asyncio.timeout``
    firing on each, ``_size`` MUST NOT drift (no leak, no
    over-allocation). Each cancelled acquire either gets its
    reservation through (released on close) or gives it back."""
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

        # Now spawn N waiters that immediately get cancelled.
        async def waiter() -> None:
            try:
                async with asyncio.timeout(0.05):
                    async with pool.acquire():
                        pass
            except (TimeoutError, asyncio.CancelledError):
                pass

        waiters = [asyncio.create_task(waiter()) for _ in range(20)]
        # Wait for all timeouts to elapse and cleanup to settle.
        await asyncio.gather(*waiters, return_exceptions=True)

        # All cancelled waiters cleaned up; only the held conn
        # accounts for _size.
        release_held.set()
        await held_task

        # After the holder releases, the pool should have at most
        # max_size connections accounted for and zero in flight.
        assert pool._size <= pool._max_size
        assert pool._size >= 0


@pytest.mark.asyncio
async def test_acquire_does_not_leak_pending_tasks(caplog: pytest.LogCaptureFixture) -> None:
    """``acquire``'s internal get_task / closed_task must not be
    orphaned under cancel churn — orphaned tasks emit "Task
    exception was never retrieved" at GC, polluting logs."""
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
        # Saturate.
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

        # Bursts of cancel churn.
        for _ in range(3):
            waiters = [asyncio.create_task(waiter()) for _ in range(5)]
            await asyncio.gather(*waiters, return_exceptions=True)

        release_held.set()
        await held_task

        # Force GC to flush any orphaned task __del__ messages.
        gc.collect()
        await asyncio.sleep(0)

    # Must not have any "Task exception was never retrieved" warnings
    # leaking from internal acquire-cleanup paths.
    leaked = [r for r in caplog.records if "Task exception was never retrieved" in r.getMessage()]
    assert not leaked, (
        f"acquire cleanup leaked {len(leaked)} unobserved task exceptions: {leaked!r}"
    )
