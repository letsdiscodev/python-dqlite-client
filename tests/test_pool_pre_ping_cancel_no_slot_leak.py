"""Pin: cancellation landing during the pre-ping path's shielded
``conn.close()`` does NOT leak a reservation slot.

The pre-ping branch in ``ConnectionPool.acquire`` shields the dead-
connection close with ``asyncio.shield`` so the close completes
even if the outer caller is cancelled. The drain-idle and create
calls each have their own ``except BaseException`` arms that release
the reservation. This test pins the cumulative invariant: under
rapid cancellation during the pre-ping path, ``pool._size`` does
not drift above ``pool._max_size``, and a follow-up ``acquire``
still works.

The current implementation is correct (verified by tracing each
exception path); this test guards against a refactor that re-orders
the close/drain/create sequence and silently breaks the
no-leak invariant.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _alive_conn(*, dead: bool = False, slow_close: bool = False) -> MagicMock:
    """Build a MagicMock that mimics the slice of ``DqliteConnection``
    the pool's acquire flow touches. ``slow_close=True`` makes
    ``close()`` await once so a cancel can land during the shielded
    body."""
    conn = MagicMock()
    conn.is_connected = True
    if slow_close:

        async def _slow_close() -> None:
            await asyncio.sleep(0)

        conn.close = AsyncMock(side_effect=_slow_close)
    else:
        conn.close = AsyncMock()
    conn._pool_released = False
    proto = MagicMock()
    proto.is_wire_coherent = True
    transport = MagicMock()
    transport.is_closing.return_value = dead
    proto._writer = MagicMock()
    proto._writer.transport = transport
    proto._reader = MagicMock()
    proto._reader.at_eof.return_value = False
    conn._protocol = proto
    return conn


@pytest.mark.asyncio
async def test_pre_ping_cancel_during_close_no_slot_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel landing during the shielded ``conn.close()`` of the
    dead-conn path must release the reservation. After the cancel,
    pool size must be back to a state where new acquires succeed
    without exceeding max_size."""
    pool = ConnectionPool(["a:9001"], min_size=0, max_size=1)
    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", AsyncMock())
    pool._initialized = True
    dead = _alive_conn(dead=True, slow_close=True)
    pool._pool.put_nowait(dead)
    pool._size = 1

    fresh = _alive_conn(dead=False)

    async def fake_create(self: object) -> MagicMock:
        return fresh

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool._create_connection", fake_create)

    async def caller() -> None:
        async with pool.acquire():
            pass

    task = asyncio.create_task(caller())
    # Yield enough times for the task to enter the pre-ping shield.
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # Reservation must not leak.
    assert pool._size <= pool._max_size, (
        f"reservation leaked under cancel: _size={pool._size} > max_size={pool._max_size}"
    )

    # And the pool must still be acquireable.
    async with pool.acquire() as conn:
        assert conn is not None


@pytest.mark.asyncio
async def test_pre_ping_cancel_after_close_before_drain_no_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel landing during ``_drain_idle`` (after the shielded close
    completes) must also release the reservation."""
    pool = ConnectionPool(["a:9001"], min_size=0, max_size=1)
    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", AsyncMock())
    pool._initialized = True
    dead = _alive_conn(dead=True)  # close completes synchronously
    pool._pool.put_nowait(dead)
    pool._size = 1

    # _drain_idle is the first await after close; cancel during its
    # await releases via the except BaseException arm.
    async def slow_drain(self: object) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool._drain_idle", slow_drain)

    fresh = _alive_conn(dead=False)

    async def fake_create(self: object) -> MagicMock:
        return fresh

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool._create_connection", fake_create)

    async def caller() -> None:
        async with pool.acquire():
            pass

    task = asyncio.create_task(caller())
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert pool._size <= pool._max_size
    async with pool.acquire() as conn:
        assert conn is not None
