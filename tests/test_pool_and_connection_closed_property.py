"""ConnectionPool.closed and DqliteConnection.closed public properties.

closed is distinct from is_connected: never-connected is closed=False/
is_connected=False; connected is False/True; closed is True/False."""

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_closed_initially_false() -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    assert pool.closed is False


@pytest.mark.asyncio
async def test_pool_closed_after_close() -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    await pool.close()
    assert pool.closed is True


@pytest.mark.asyncio
async def test_pool_closed_idempotent() -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    await pool.close()
    await pool.close()
    assert pool.closed is True


def test_connection_closed_initially_false() -> None:
    conn = DqliteConnection("h:9001")
    assert conn.closed is False
    assert conn.is_connected is False


@pytest.mark.asyncio
async def test_connection_closed_after_close_without_connect() -> None:
    """A never-connected DqliteConnection still flips closed to True after
    close() despite the never-connected short-circuit."""
    conn = DqliteConnection("h:9001")
    await conn.close()
    assert conn.closed is True
    assert conn.is_connected is False
