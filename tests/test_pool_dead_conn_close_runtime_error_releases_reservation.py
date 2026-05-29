"""Pin: ``acquire``'s dead-conn close must absorb ``RuntimeError`` (raised by
``DqliteConnection.close()`` on a closed/cross-loop event loop) and release the
reservation. An escape here leaks one slot, skipping the cleanup arms below."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _make_dead_looking_conn() -> MagicMock:
    # ``not is_connected`` short-circuits before ``_socket_looks_dead``.
    conn = MagicMock()
    conn.is_connected = False
    conn._pool_released = True  # set by ``_release`` on enqueue
    conn._protocol = None
    return conn


@pytest.mark.asyncio
async def test_dead_conn_close_runtime_error_releases_reservation() -> None:
    """A RuntimeError from the dead-conn close must not escape acquire() and
    must release the reservation, leaving _size at baseline (no leak)."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=2, timeout=1.0)

    queued = _make_dead_looking_conn()

    async def _boom_close() -> None:
        raise RuntimeError("Event loop is closed")

    queued.close = AsyncMock(side_effect=_boom_close)
    pool._pool.put_nowait(queued)
    pool._size = 1  # reservation slot for the queued conn

    # Short-circuit the dial; the assertion is reservation accounting, not the
    # create outcome.
    async def _fail_create() -> object:
        raise RuntimeError("test: create_connection short-circuited")

    pool._create_connection = AsyncMock(side_effect=_fail_create)

    size_before = pool._size
    with pytest.raises(RuntimeError):
        async with pool.acquire():
            pytest.fail("acquire should not yield a connection")
    size_after = pool._size

    assert size_after <= size_before - 1, (
        f"dead-conn close RuntimeError must release the reservation: "
        f"size_before={size_before}, size_after={size_after}"
    )
