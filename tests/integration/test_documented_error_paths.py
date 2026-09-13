"""Live coverage for the error paths docs/architecture.md promises."""

from __future__ import annotations

import gc
import os

import pytest

from dqliteclient import (
    ClusterClient,
    ClusterPolicyError,
    DataError,
    DqliteConnection,
    InterfaceError,
    MemoryNodeStore,
    ProtocolError,
    connect,
)

pytestmark = pytest.mark.integration


async def test_unencodable_parameter_is_a_data_error(cluster_address: str) -> None:
    async with DqliteConnection(cluster_address) as conn:
        with pytest.raises(DataError):
            await conn.fetchall("SELECT ?", [object()])
        assert conn.is_connected  # a client-side encode failure leaves the session usable
        assert await conn.fetchval("SELECT 1") == 1


async def test_redirect_policy_rejection_is_not_retried(cluster_address: str) -> None:
    cluster = ClusterClient(MemoryNodeStore([cluster_address]), redirect_policy=lambda a: False)
    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()
    with pytest.raises(ClusterPolicyError):
        await cluster.connect(max_attempts=3)


async def test_row_cap_invalidates_the_connection(cluster_address: str) -> None:
    async with DqliteConnection(cluster_address) as setup:
        await setup.execute("DROP TABLE IF EXISTS test_row_cap")
        await setup.execute("CREATE TABLE test_row_cap (n INTEGER PRIMARY KEY)")
        for n in range(10):
            await setup.execute("INSERT INTO test_row_cap (n) VALUES (?)", [n])
    conn = await connect(cluster_address, max_total_rows=5)
    try:
        with pytest.raises(ProtocolError, match="max_total_rows"):
            await conn.fetchall("SELECT n FROM test_row_cap")
        assert not conn.is_connected
    finally:
        await conn.close()
    async with DqliteConnection(cluster_address, max_total_rows=10) as ok:
        assert len(await ok.fetchall("SELECT n FROM test_row_cap")) == 10


async def test_fork_guard(cluster_address: str, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = await connect(cluster_address)
    try:
        real_pid = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
        with pytest.raises(InterfaceError, match="used after fork"):
            await conn.execute("SELECT 1")
    finally:
        monkeypatch.undo()
        await conn.close()


async def test_unclosed_connection_warns(cluster_address: str) -> None:
    conn = await connect(cluster_address)
    with pytest.warns(ResourceWarning, match="garbage-collected"):
        del conn
        gc.collect()
