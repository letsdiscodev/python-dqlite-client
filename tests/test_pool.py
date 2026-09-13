"""ConnectionPool: contract, initialize failure reporting, acquire timeouts, fork guard, sizing."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError, InterfaceError, OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


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


@pytest.mark.asyncio
async def test_fresh_slot_create_clamp_translates_at_clamp_scope() -> None:
    """A fresh-slot reservation with a slow ``_create_connection`` must
    surface as ``DqliteConnectionError("Timed out ...") from TimeoutError``."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.05)

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied")

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with pool.acquire():
            pytest.fail("should not reach")

    err = exc_info.value
    assert "Timed out creating a fresh connection from the pool" in str(err)
    assert isinstance(err.__cause__, TimeoutError), (
        f"expected DqliteConnectionError chained from TimeoutError; got __cause__={err.__cause__!r}"
    )


@pytest.mark.asyncio
async def test_already_expired_deadline_carries_actionable_cause_text() -> None:
    """When the deadline is already past before the clamp scope opens, the
    ``DqliteConnectionError`` is chained from a TimeoutError naming the
    overshoot. Either branch may land given timing; the load-bearing pin is
    that a TimeoutError is present in the chain."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.001)

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied")

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with pool.acquire():
            pytest.fail("should not reach")

    err = exc_info.value
    assert isinstance(err.__cause__, TimeoutError)


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


@pytest.mark.asyncio
async def test_pool_initialize_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=1, max_size=2)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_pool_acquire_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        async with pool.acquire():
            pytest.fail("should not reach")


def test_operational_error_raw_message_capped_bounds_pickle_size() -> None:
    """A 64 KiB raw_message pickles to ~5 KiB rather than ~70 KiB."""
    hostile = "X" * 64_000
    e = OperationalError(hostile, 1, raw_message=hostile)
    pickled = pickle.dumps(e)
    assert len(pickled) < 10_000, (
        f"OperationalError pickle size {len(pickled)} bytes — raw_message cap regression"
    )


def test_aggregate_exception_group_payload_scales_with_min_size_not_unbounded() -> None:
    """A group of 10 capped OperationalErrors pickles to ~50 KB, not ~640 KB."""
    hostile = "Y" * 64_000
    failures = [OperationalError(hostile, 1, raw_message=hostile) for _ in range(10)]
    group = BaseExceptionGroup("pool.initialize: 10 of 10 connects failed", failures)
    pickled = pickle.dumps(group)
    assert len(pickled) < 100_000, (
        f"BaseExceptionGroup pickle size {len(pickled)} bytes — "
        f"per-exception raw_message cap not enforcing aggregate bound"
    )


@pytest.mark.asyncio
async def test_initialize_single_failure_raises_narrow_type(monkeypatch) -> None:
    """One failure re-raises the narrow exception type."""
    pool = ConnectionPool(
        ["localhost:9001"],
        min_size=1,
        max_size=1,
        timeout=0.5,
    )

    async def _fail() -> Any:
        raise DqliteConnectionError("refused")

    monkeypatch.setattr(pool, "_create_connection", _fail)

    with pytest.raises(DqliteConnectionError, match="refused"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_initialize_multiple_failures_raises_exception_group(monkeypatch, caplog) -> None:
    """Three failures raise a ``BaseExceptionGroup``, each logged at WARNING."""
    pool = ConnectionPool(
        ["a:9001", "b:9001", "c:9001"],
        min_size=3,
        max_size=3,
        timeout=0.5,
    )

    failures_iter = iter(
        [
            TimeoutError("timeout"),
            DqliteConnectionError("refused"),
            ConnectionError("peer rst"),
        ]
    )

    async def _fail() -> Any:
        raise next(failures_iter)

    monkeypatch.setattr(pool, "_create_connection", _fail)

    caplog.set_level(logging.WARNING, logger="dqliteclient.pool")
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await pool.initialize()
    eg = exc_info.value
    assert len(eg.exceptions) == 3
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("create_connection" in m for m in warning_msgs) == 3


def test_pool_min_size_rejects_bool() -> None:
    with pytest.raises(TypeError, match="min_size must be int"):
        ConnectionPool(addresses=["localhost:9001"], min_size=True)


def test_pool_max_size_rejects_bool() -> None:
    with pytest.raises(TypeError, match="max_size must be int"):
        ConnectionPool(addresses=["localhost:9001"], max_size=True)


def test_pool_max_attempts_rejects_bool() -> None:
    with pytest.raises(TypeError, match="max_attempts must be int"):
        ConnectionPool(
            addresses=["localhost:9001"],
            max_attempts=True,
        )


@pytest.mark.asyncio
async def test_cluster_connect_max_attempts_rejects_bool() -> None:
    cluster = ClusterClient(MemoryNodeStore())
    with pytest.raises(TypeError, match="max_attempts must be int"):
        await cluster.connect(database="x", max_attempts=True)
