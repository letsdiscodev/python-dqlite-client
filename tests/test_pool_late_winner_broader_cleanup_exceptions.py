"""``_put_back_or_release_late_winner``'s two close arms suppress the full
``_POOL_CLEANUP_EXCEPTIONS`` tuple, not just ``OSError``; an escaped exception
skips ``_release_reservation`` and leaks the slot permanently."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import (
    DqliteConnectionError,
    InterfaceError,
)
from dqliteclient.pool import ConnectionPool


def _make_conn_close_raises(exc: BaseException) -> MagicMock:
    conn = MagicMock()
    conn._pool_released = True

    async def _close() -> None:
        # Mirror DqliteConnection.close: pool-released conns short-circuit.
        if conn._pool_released:
            return
        raise exc

    conn.close = AsyncMock(side_effect=_close)
    return conn


@pytest.mark.asyncio
async def test_closed_pool_arm_suppresses_dqlite_connection_error() -> None:
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1

    conn = _make_conn_close_raises(DqliteConnectionError("peer reset"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 0, (
        f"closed-pool arm leaked the reservation slot when close() "
        f"raised DqliteConnectionError; _size={pool._size} (want 0)"
    )


@pytest.mark.asyncio
async def test_closed_pool_arm_suppresses_interface_error() -> None:
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1

    conn = _make_conn_close_raises(InterfaceError("owned by another task"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 0, (
        f"closed-pool arm leaked the reservation slot when close() "
        f"raised InterfaceError; _size={pool._size} (want 0)"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_suppresses_dqlite_connection_error() -> None:
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    pool._pool.put_nowait(MagicMock())
    pool._size = 2

    conn = _make_conn_close_raises(DqliteConnectionError("peer reset"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 1, (
        f"QueueFull arm leaked the reservation slot when close() "
        f"raised DqliteConnectionError; _size={pool._size} (want 1)"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_suppresses_interface_error() -> None:
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    pool._pool.put_nowait(MagicMock())
    pool._size = 2

    conn = _make_conn_close_raises(InterfaceError("owned by another task"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 1, (
        f"QueueFull arm leaked the reservation slot when close() "
        f"raised InterfaceError; _size={pool._size} (want 1)"
    )
