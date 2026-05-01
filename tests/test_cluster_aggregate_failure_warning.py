"""Pin: cluster-wide unreachable / connect-exhaustion log a single
WARNING summary at the aggregate-failure decision point — distinct
from the per-attempt DEBUG noise that fires on every individual
probe / retry.

Per-attempt log lines stay at DEBUG (a routine leader flip's
per-attempt churn must not spam logs at default verbosity), but the
all-attempts-exhausted outcome is the one event paged operators need
to see at default verbosity. Without this, operators tailing logs at
INFO see nothing during the failure cascade and only the
application-level traceback after the caller catches the exception.
"""

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
    """When every node in the cluster fails leader discovery, an
    aggregate WARNING fires before the ClusterError raise."""
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
    """Per-attempt failures must NOT escalate to WARNING — those would
    spam logs during routine leader-flip churn. The aggregate WARNING
    is the only WARNING that fires."""
    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=0.5)
    cluster._query_leader = AsyncMock(side_effect=DqliteConnectionError("connection refused"))

    with pytest.raises(ClusterError):
        await cluster.find_leader()

    # Per-node DEBUG lines should be present; per-node WARNING lines
    # should not.
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
