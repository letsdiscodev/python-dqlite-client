"""Pin: ``ConnectionPool.acquire`` peeks the transport via
``_socket_looks_dead`` *before* yielding an idle connection to the
caller, so a connection that saw a clean peer FIN (leader flip with
graceful close) does not get handed out as a zombie.

The protocol-level ``is_connected`` is just ``self._protocol is not
None`` — it does NOT detect a half-closed socket. Without the
transport-level peek, the user's first query after a leader flip
fails on a dequeue path that the pool could have self-healed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _alive_conn(*, dead: bool = False) -> MagicMock:
    """Build a MagicMock that mimics the slice of ``DqliteConnection``
    the pool's acquire flow touches.

    With ``dead=True``, the transport peek tells ``_socket_looks_dead``
    the connection is gone (transport.is_closing() returns True).
    """
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
    """An idle connection whose transport is closing must not be
    yielded to the caller. The pool drains the idle queue and creates
    a fresh connection."""
    pool = ConnectionPool(["a:9001"], min_size=0, max_size=1)
    # Skip real cluster bootstrap; we only exercise the acquire path.
    monkeypatch.setattr("dqliteclient.pool.ConnectionPool.initialize", AsyncMock())
    pool._initialized = True
    # Seed one dead conn into the idle queue and reserve its slot.
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
    """Regression pin: a healthy idle connection passes the peek and
    is yielded unchanged — the pre-ping does NOT force unnecessary
    reconnects."""
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

    # Avoid leaking the asyncio queue close-task on test teardown
    await asyncio.sleep(0)
