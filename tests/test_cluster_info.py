"""ClusterClient.cluster_info: policy filtering and transient raft errors."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.fixture
def cluster_with_returned_nodes():
    """Cluster client whose cluster_info() returns three nodes."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [
        NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="bystander:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=3, address="hostile:9001", role=NodeRole.SPARE),
    ]

    fake_proto = MagicMock()
    fake_proto.cluster = AsyncMock(return_value=nodes)
    # Same address as leader_addr selects the no-flip happy path.
    fake_proto.get_leader = AsyncMock(return_value=(1, "leader:9001"))

    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cluster.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    return cluster


@pytest.mark.asyncio
async def test_no_policy_returns_unfiltered(cluster_with_returned_nodes) -> None:
    nodes = await cluster_with_returned_nodes.cluster_info()
    assert len(nodes) == 3


@pytest.mark.asyncio
async def test_policy_filters_rejected_nodes(cluster_with_returned_nodes) -> None:
    def reject_hostile(addr: str) -> bool:
        return "hostile" not in addr

    nodes = await cluster_with_returned_nodes.cluster_info(policy=reject_hostile)
    assert {n.address for n in nodes} == {"leader:9001", "bystander:9001"}


@pytest.mark.asyncio
async def test_policy_rejection_logs_warning(
    cluster_with_returned_nodes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def reject_hostile(addr: str) -> bool:
        return "hostile" not in addr

    with caplog.at_level(logging.WARNING):
        await cluster_with_returned_nodes.cluster_info(policy=reject_hostile)
    assert any("hostile:9001" in r.message for r in caplog.records)
    assert any("rejected by policy" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_instance_redirect_policy_used_when_no_per_call_policy() -> None:
    """No per-call policy falls back to the instance-level redirect_policy."""

    def reject_hostile(addr: str) -> bool:
        return "hostile" not in addr

    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        redirect_policy=reject_hostile,
    )
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [
        NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=3, address="hostile:9001", role=NodeRole.SPARE),
    ]
    fake_proto = MagicMock()
    fake_proto.cluster = AsyncMock(return_value=nodes)
    fake_proto.get_leader = AsyncMock(return_value=(1, "leader:9001"))
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cluster.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    filtered = await cluster.cluster_info()
    assert {n.address for n in filtered} == {"leader:9001"}


@pytest.mark.asyncio
async def test_per_call_policy_overrides_instance_policy() -> None:
    """A per-call policy overrides the instance redirect_policy, even a permissive one."""

    def reject_everything(_addr: str) -> bool:
        return False

    def accept_everything(_addr: str) -> bool:
        return True

    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        redirect_policy=reject_everything,
    )
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    fake_proto = MagicMock()
    fake_proto.cluster = AsyncMock(return_value=nodes)
    fake_proto.get_leader = AsyncMock(return_value=(1, "leader:9001"))
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cluster.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    filtered = await cluster.cluster_info(policy=accept_everything)
    assert len(filtered) == 1
    cluster_default = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        redirect_policy=accept_everything,
    )
    cluster_default.find_leader = AsyncMock(return_value="leader:9001")
    cluster_default.open_admin_connection = MagicMock(return_value=fake_admin_cm)
    filtered = await cluster_default.cluster_info(policy=reject_everything)
    assert len(filtered) == 0


def _make_admin_cm(proto: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=proto)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_raft_nomem_transient_falls_back_to_local_responder() -> None:
    """(N, "") RAFT_NOMEM transient: read config from the current responder."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(99, ""))  # RAFT_NOMEM: id, no address
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    result = await cluster.cluster_info()

    assert result == nodes
    proto.cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_no_leader_known_sentinel_falls_back_to_local_responder() -> None:
    """Canonical (0, "") sentinel also falls back to the local responder."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(0, ""))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    result = await cluster.cluster_info()

    assert result == nodes
