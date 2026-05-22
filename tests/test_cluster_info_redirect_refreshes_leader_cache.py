"""Pin: ``cluster_info``'s redirect-verify success path refreshes the
last-known-leader cache to the verified address.

Sibling ``leader_info`` updates the cache when its redirect-verify
arm confirms a new leader; ``cluster_info`` previously left the cache
either ``None`` (cleared upstream by ``find_leader``'s fast-path
miss) or stale (pointing at the responder that just stepped down).

The next ``find_leader`` then either misses the fast path entirely
(extra full sweep) or hits the stale entry (one extra wasted probe).
Performance regression specifically on the post-leader-flip path —
the very scenario where the leader cache matters most.
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
    """Mid-flip path: the responder hands back a different leader
    address; ``_verify_redirect`` confirms it; the cluster RPC lands
    on the verified leader. After the call, the last-known-leader
    cache MUST equal the verified address so the next
    ``find_leader`` hits the fast path.
    """
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

    # Pre-condition: a stale cache value distinct from the verified
    # leader. The post-condition checks that the value was overwritten.
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
    """On the no-flip happy path the cache stays warm: the responder
    just answered the RPC and is provably the leader. The success
    arm of ``cluster_info`` does NOT touch the cache (mirroring
    go-dqlite's ``Client.Cluster``).
    """
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._set_last_known_leader("leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(1, "leader:9001"))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    await cluster.cluster_info()
    # Cache stays warm at the original (still-current) leader.
    assert cluster._get_last_known_leader() == "leader:9001"
