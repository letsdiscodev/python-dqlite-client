"""``transfer_leadership`` and ``remove_node`` invalidate the
``_last_known_leader`` cache. Without this, the next ``find_leader``
takes the fast path and probes the now-stale leader (one wasted RTT),
falls through, and only then re-discovers via the full sweep.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster_with_cached_leader(addr: str = "node1:9001") -> ClusterClient:
    cluster = ClusterClient(MemoryNodeStore([addr]), timeout=2.0)
    cluster._set_last_known_leader(addr)
    return cluster


def _patch_admin(cluster: ClusterClient, fake_proto: MagicMock):
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object):
        return reader, writer

    return [
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ]


@pytest.mark.asyncio
async def test_transfer_leadership_invalidates_last_known_leader() -> None:
    cluster = _make_cluster_with_cached_leader()
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.transfer = AsyncMock()

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        await cluster.transfer_leadership(target_node_id=2)
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_remove_node_invalidates_last_known_leader() -> None:
    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.remove = AsyncMock()

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        await cluster.remove_node(node_id=2)
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_add_node_invalidates_last_known_leader() -> None:
    """Adding a new VOTER changes quorum size; the new majority can
    elect a different leader during commit. Invalidate the cache so
    the next find_leader runs the full sweep rather than paying a
    fast-path miss against the (possibly former) leader."""
    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.add = AsyncMock()
    fake_proto.assign = AsyncMock()

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        await cluster.add_node(node_id=42, address="node42:9001")
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_assign_role_invalidates_last_known_leader() -> None:
    """Promoting STANDBY → VOTER widens the voting set; the
    election window applies. Invalidate the cache."""
    from dqlitewire import NodeRole

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.assign = AsyncMock()

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        await cluster.assign_role(node_id=2, role=NodeRole.VOTER)
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_set_weight_invalidates_last_known_leader() -> None:
    """Changing weight does not directly change quorum, but a
    marginal cluster on the edge of an election can be tipped by
    the weight shift. Invalidate the cache."""
    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.weight = AsyncMock()

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        await cluster.set_weight(weight=5)
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_transfer_leadership_invalidates_cache_on_failure() -> None:
    """Pin: a leader-flip-induced failure during admin RPC must also
    invalidate the cache. Previously the success-path-only invalidation
    left the cache pointing at the rejecter for one wasted RTT on the
    next ``find_leader``."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.transfer = AsyncMock(
        side_effect=OperationalError("not leader", code=1001, raw_message="not leader")
    )

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.transfer_leadership(target_node_id=2)
    finally:
        for p in reversed(patches):
            p.stop()

    # Cache invalidated even though the RPC failed.
    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_add_node_invalidates_cache_on_failure() -> None:
    """add_node failure (e.g. ASSIGN second-phase fails after ADD
    landed) must also invalidate the cache."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.add = AsyncMock()  # ADD succeeds
    fake_proto.assign = AsyncMock(
        side_effect=OperationalError("not leader", code=1001, raw_message="not leader")
    )

    from dqlitewire import NodeRole

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.add_node(node_id=2, address="node2:9001", role=NodeRole.VOTER)
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None
