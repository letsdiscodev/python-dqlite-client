"""``cluster_info``'s redirect-verify success path refreshes the last-known-leader
cache to the verified address, so the next find_leader hits the fast path on the
post-flip scenario where the cache matters most.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _make_admin_cm(proto: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=proto)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_cluster_info_redirect_verify_success_refreshes_leader_cache() -> None:
    """Mid-flip: after verifying the new leader, the cache holds the verified address."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._check_redirect = MagicMock(return_value=None)
    cluster._verify_redirect = AsyncMock(return_value="new-leader:9001")

    nodes_from_verified = [
        NodeInfo(node_id=2, address="new-leader:9001", role=NodeRole.VOTER),
    ]

    stale_proto = MagicMock()
    stale_proto.get_leader = AsyncMock(return_value=(2, "new-leader:9001"))
    stale_proto.cluster = AsyncMock(
        return_value=[NodeInfo(node_id=1, address="leader:9001", role=NodeRole.SPARE)]
    )

    verified_proto = MagicMock()
    verified_proto.cluster = AsyncMock(return_value=nodes_from_verified)

    admin_cms = [_make_admin_cm(stale_proto), _make_admin_cm(verified_proto)]
    cluster.open_admin_connection = MagicMock(side_effect=admin_cms)

    # Stale cache value, distinct from the verified leader, must be overwritten.
    cluster._set_last_known_leader("stale-cache:9001")

    result = await cluster.cluster_info()
    assert result == nodes_from_verified

    cached = cluster._get_last_known_leader()
    assert cached == "new-leader:9001", (
        "cluster_info's redirect-verify success path must refresh "
        "_last_known_leader to the verified address so the next "
        "find_leader hits the fast path; got "
        f"{cached!r}"
    )


@pytest.mark.asyncio
async def test_cluster_info_no_flip_does_not_clobber_leader_cache() -> None:
    """No-flip: the success arm does not touch the warm cache (matches go-dqlite)."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._set_last_known_leader("leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(1, "leader:9001"))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    await cluster.cluster_info()
    assert cluster._get_last_known_leader() == "leader:9001"
