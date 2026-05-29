"""``ConnectionPool.acquire`` peeks the transport via ``_socket_looks_dead`` before
yielding an idle connection, since ``is_connected`` (``self._protocol is not None``)
does not detect a half-closed socket after a clean peer FIN."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _alive_conn(*, dead: bool = False) -> MagicMock:
    """Mock the slice of ``DqliteConnection`` acquire touches; ``dead=True`` makes
    transport.is_closing() return True so ``_socket_looks_dead`` sees it gone."""
    conn = MagicMock()
    conn.is_connected = True
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
async def test_acquire_drops_dead_idle_connection_and_returns_fresh_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle connection whose transport is closing must not be yielded; the pool
    drains the queue and creates a fresh connection."""
    pool = ConnectionPool(["a:9001"], min_size=0, max_size=1)
    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", AsyncMock())
    pool._initialized = True
    dead = _alive_conn(dead=True)
    pool._pool.put_nowait(dead)
    pool._size = 1

    fresh = _alive_conn(dead=False)
    create_calls = 0

    async def fake_create(self: object) -> MagicMock:
        nonlocal create_calls
        create_calls += 1
        return fresh

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool._create_connection", fake_create)

    async with pool.acquire() as conn:
        assert conn is fresh, "acquire must drop the dead idle connection and yield a fresh one"
        assert dead.close.called, "dead connection must have been closed"
    assert create_calls == 1


@pytest.mark.asyncio
async def test_acquire_yields_alive_idle_connection_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy idle connection passes the peek and is yielded unchanged."""
    pool = ConnectionPool(["a:9001"], min_size=0, max_size=1)
    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", AsyncMock())
    pool._initialized = True
    alive = _alive_conn(dead=False)
    pool._pool.put_nowait(alive)
    pool._size = 1

    create_calls = 0

    async def fake_create(self: object) -> MagicMock:
        nonlocal create_calls
        create_calls += 1
        return _alive_conn()

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool._create_connection", fake_create)

    async with pool.acquire() as conn:
        assert conn is alive, "alive idle connection must pass through unchanged"
        assert not alive.close.called
    assert create_calls == 0, "no fresh create when the idle connection passes the peek"


@pytest.mark.asyncio
async def test_acquire_peeks_via_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_socket_looks_dead`` triggers on EOF too, not just
    transport.is_closing()."""
    pool = ConnectionPool(["a:9001"], min_size=0, max_size=1)
    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", AsyncMock())
    pool._initialized = True
    eof_conn = _alive_conn(dead=False)
    eof_conn._protocol._reader.at_eof.return_value = True
    pool._pool.put_nowait(eof_conn)
    pool._size = 1

    fresh = _alive_conn()

    async def fake_create(self: object) -> MagicMock:
        return fresh

    monkeypatch.setattr("dqliteclient.pool.ConnectionPool._create_connection", fake_create)

    async with pool.acquire() as conn:
        assert conn is fresh
        assert eof_conn.close.called

    # Drain the queue close-task so it doesn't leak on teardown.
    await asyncio.sleep(0)
