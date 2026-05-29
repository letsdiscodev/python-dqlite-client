"""Connect exhaustion logs a single aggregate WARNING; per-attempt lines stay
at DEBUG so routine leader-flip churn does not spam logs at default verbosity."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_find_leader_logs_aggregate_warning_on_all_nodes_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An aggregate WARNING fires before the ClusterError when all nodes fail."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=0.5)
    cluster._query_leader = AsyncMock(side_effect=DqliteConnectionError("connection refused"))

    with pytest.raises(ClusterError):
        await cluster.find_leader()

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dqliteclient.cluster"
    ]
    assert any("leader discovery failed" in r.getMessage() for r in warnings), (
        "Aggregate WARNING must fire when all nodes fail leader discovery; "
        f"saw warnings={[r.getMessage() for r in warnings]}"
    )


@pytest.mark.asyncio
async def test_per_node_failures_remain_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Per-attempt failures stay at DEBUG; the aggregate is the only WARNING."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=0.5)
    cluster._query_leader = AsyncMock(side_effect=DqliteConnectionError("connection refused"))

    with pytest.raises(ClusterError):
        await cluster.find_leader()

    debug_lines = [
        r for r in caplog.records if r.levelno == logging.DEBUG and r.name == "dqliteclient.cluster"
    ]
    warning_lines = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dqliteclient.cluster"
    ]
    assert any("find_leader" in r.getMessage() for r in debug_lines)
    assert len(warning_lines) == 1, (
        f"Expected exactly one aggregate WARNING; got {len(warning_lines)}: "
        f"{[r.getMessage() for r in warning_lines]}"
    )
