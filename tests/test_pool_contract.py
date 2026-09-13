"""Pool contract with stubbed connections: reuse, dead-connection replacement, reset, close."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dqliteclient import ConnectionPool, DqliteConnection, DqliteConnectionError, OperationalError


class Factory:
    def __init__(self, connected: Any) -> None:
        self._connected = connected
        self.created: list[tuple[DqliteConnection, Any]] = []
        self.fail: Exception | None = None

    async def __call__(self) -> DqliteConnection:
        if self.fail is not None:
            raise self.fail
        conn, proto = self._connected()
        self.created.append((conn, proto))
        assert isinstance(conn, DqliteConnection)
        return conn


@pytest.fixture
def make_pool(monkeypatch: pytest.MonkeyPatch, connected: Any) -> Any:
    def make(**kwargs: Any) -> tuple[ConnectionPool, Factory]:
        pool = ConnectionPool(["localhost:9001"], **kwargs)
        factory = Factory(connected)
        monkeypatch.setattr(pool, "_create_connection", factory)
        return pool, factory

    return make


async def test_idle_connection_is_reused(make_pool: Any) -> None:
    pool, factory = make_pool(min_size=0, max_size=2)
    async with pool.acquire() as first:
        pass
    async with pool.acquire() as second:
        pass
    assert first is second and len(factory.created) == 1
    await pool.close()
    assert first.closed


async def test_dead_idle_connection_is_replaced(make_pool: Any) -> None:
    pool, factory = make_pool(min_size=0, max_size=1)
    async with pool.acquire() as first:
        pass
    factory.created[0][1].is_alive = False
    async with pool.acquire() as second:
        pass
    assert second is not first and len(factory.created) == 2
    await pool.close()


async def test_release_rolls_back_open_transaction(make_pool: Any) -> None:
    pool, factory = make_pool(min_size=0, max_size=1)
    async with pool.acquire() as conn:
        await conn.execute("BEGIN")
    assert factory.created[0][1].sent[-1] == "ROLLBACK"
    assert conn.in_transaction is False
    async with pool.acquire() as again:
        assert again is conn
    await pool.close()


async def test_release_drops_connection_when_rollback_fails(make_pool: Any) -> None:
    pool, factory = make_pool(min_size=0, max_size=1)
    async with pool.acquire() as conn:
        await conn.execute("BEGIN")
        factory.created[0][1].fail_with["ROLLBACK"] = OperationalError("disk I/O error", code=10)
    assert conn.closed
    async with pool.acquire() as fresh:
        assert fresh is not conn
    await pool.close()


async def test_connection_invalidated_in_block_is_not_pooled(make_pool: Any) -> None:
    pool, factory = make_pool(min_size=0, max_size=1)
    async with pool.acquire() as conn:
        factory.created[0][1].fail_with["SELECT"] = DqliteConnectionError(
            "Connection closed by server"
        )
        with pytest.raises(DqliteConnectionError):
            await conn.execute("SELECT 1")
    assert conn.closed
    async with pool.acquire() as fresh:
        assert fresh is not conn
    await pool.close()


async def test_acquire_waits_for_a_slot_then_times_out(make_pool: Any) -> None:
    pool, _ = make_pool(min_size=0, max_size=1, timeout=0.05)
    async with pool.acquire():
        with pytest.raises(DqliteConnectionError, match="Timed out waiting for a connection"):
            async with pool.acquire():
                pass
    async with pool.acquire():
        pass
    await pool.close()


async def test_waiter_gets_the_slot_when_released(make_pool: Any) -> None:
    pool, _ = make_pool(min_size=0, max_size=1, timeout=1.0)
    got: list[DqliteConnection] = []

    async def waiter() -> None:
        async with pool.acquire() as conn:
            got.append(conn)

    async with pool.acquire() as held:
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        assert not got
    await task
    assert got == [held]
    await pool.close()


async def test_initialize_opens_min_size_and_failure_is_reported(make_pool: Any) -> None:
    pool, factory = make_pool(min_size=2, max_size=3)
    await pool.initialize()
    assert len(factory.created) == 2
    await pool.close()
    pool, factory = make_pool(min_size=1, max_size=3)
    factory.fail = DqliteConnectionError("refused")
    with pytest.raises(DqliteConnectionError, match="refused"):
        await pool.initialize()
    factory.fail = None
    await pool.initialize()
    assert len(factory.created) == 1
    await pool.close()


async def test_closed_pool_rejects_acquire_and_closes_late_returns(make_pool: Any) -> None:
    pool, _ = make_pool(min_size=0, max_size=2)
    async with pool.acquire() as conn:
        await pool.close()
        assert not conn.closed
    assert conn.closed
    with pytest.raises(DqliteConnectionError, match="Pool is closed"):
        async with pool.acquire():
            pass
