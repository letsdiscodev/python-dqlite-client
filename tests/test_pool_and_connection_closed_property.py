"""Pin: ``ConnectionPool.closed`` and ``DqliteConnection.closed``
public properties.

Mirrors dbapi-layer ``Connection.closed`` / psycopg / asyncpg /
aiosqlite parity — direct dqliteclient consumers (sqlalchemy-dqlite
touches the inner client connection; advanced users embed the
client layer) get the same lifecycle predicate at every layer.

``DqliteConnection.closed`` is distinct from ``is_connected``:
- never-connected: closed=False, is_connected=False
- connected:       closed=False, is_connected=True
- closed:          closed=True,  is_connected=False
"""

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
    await pool.close()  # second call: silent no-op
    assert pool.closed is True


def test_connection_closed_initially_false() -> None:
    conn = DqliteConnection("h:9001")
    assert conn.closed is False
    assert conn.is_connected is False


@pytest.mark.asyncio
async def test_connection_closed_after_close_without_connect() -> None:
    """A never-connected DqliteConnection still flips ``closed`` to
    True after ``close()``. The fork-branch / never-connected path
    in close() short-circuits but still flips the marker."""
    conn = DqliteConnection("h:9001")
    await conn.close()
    assert conn.closed is True
    # ``is_connected`` is False either way (never had a transport).
    assert conn.is_connected is False
