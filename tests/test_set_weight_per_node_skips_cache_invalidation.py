"""``set_weight(address=<non-leader>)`` must NOT invalidate ``_last_known_leader``:
the per-node form dials ``address`` directly and never touches the leader cache."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster_with_seeded_cache(cached: str) -> ClusterClient:
    cc = ClusterClient(MemoryNodeStore(["10.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader(cached)
    fake_proto = MagicMock()
    fake_proto.weight = AsyncMock(return_value=None)
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)
    return cc


@pytest.mark.asyncio
async def test_set_weight_per_node_does_not_invalidate_leader_cache() -> None:
    cached = "10.0.0.1:9001"
    cc = _make_cluster_with_seeded_cache(cached)

    await cc.set_weight(5, address="10.0.0.5:9001")

    assert cc._get_last_known_leader() == cached, (
        "per-node set_weight must NOT invalidate _last_known_leader; the "
        "RPC dialed an explicit address and never touched the leader "
        "cache. Sibling describe() gates the invalidation correctly."
    )


@pytest.mark.asyncio
async def test_set_weight_leader_targeted_preserves_cache_on_success() -> None:
    """``address=None`` keeps the cache warm on SUCCESS: the responding leader
    has provably just answered the RPC. Matches go-dqlite's ``Client.Weight``."""
    cached = "10.0.0.1:9001"
    cc = _make_cluster_with_seeded_cache(cached)
    cc.find_leader = AsyncMock(return_value="10.0.0.1:9001")

    await cc.set_weight(5)

    assert cc._get_last_known_leader() == cached


@pytest.mark.asyncio
async def test_set_weight_per_node_does_not_invalidate_on_failure() -> None:
    """Even when the per-node RPC raises mid-call, the leader cache stays intact:
    the failure is on the targeted node, not on the leader path."""
    cached = "10.0.0.1:9001"
    cc = ClusterClient(MemoryNodeStore(["10.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader(cached)
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(side_effect=ConnectionResetError("boom"))
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    with pytest.raises(ConnectionResetError):
        await cc.set_weight(5, address="10.0.0.5:9001")

    assert cc._get_last_known_leader() == cached
