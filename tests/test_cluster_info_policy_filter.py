"""``cluster_info(policy=...)`` filters returned nodes through the
supplied predicate so a control-plane that auto-rotates membership
based on the result cannot be steered to attacker-controlled
addresses by a hostile leader.

Without the kwarg, the only protection at the higher layer was the
caller's own filter logic. Pair with default_safe_redirect_policy /
allowlist_policy.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.fixture
def cluster_with_returned_nodes():
    """Build a cluster client whose cluster_info() returns three nodes."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [
        NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="bystander:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=3, address="hostile:9001", role=NodeRole.SPARE),
    ]

    fake_proto = MagicMock()
    fake_proto.cluster = AsyncMock(return_value=nodes)

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
    """When no per-call ``policy`` is provided, ``cluster_info`` falls
    back to the instance-level ``redirect_policy`` configured at
    construction. Mirrors the precedence used by ``leader_info`` and
    by ``find_leader``'s redirect arms."""

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
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cluster.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    # Caller passes no per-call policy; instance default kicks in.
    filtered = await cluster.cluster_info()
    assert {n.address for n in filtered} == {"leader:9001"}


@pytest.mark.asyncio
async def test_per_call_policy_overrides_instance_policy() -> None:
    """A per-call ``policy`` overrides the instance-level
    ``redirect_policy``, including a deliberately permissive override."""

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
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cluster.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    # Per-call accept-everything overrides the instance reject-everything.
    filtered = await cluster.cluster_info(policy=accept_everything)
    assert len(filtered) == 1
    # Per-call reject-everything also exercised — instance is the same shape;
    # we want to be sure the per-call WINS regardless.
    cluster_default = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        redirect_policy=accept_everything,
    )
    cluster_default.find_leader = AsyncMock(return_value="leader:9001")
    cluster_default.open_admin_connection = MagicMock(return_value=fake_admin_cm)
    filtered = await cluster_default.cluster_info(policy=reject_everything)
    assert len(filtered) == 0
