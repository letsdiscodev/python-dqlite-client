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
