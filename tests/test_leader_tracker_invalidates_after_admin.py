"""Membership-mutating admin RPCs invalidate ``_last_known_leader`` so the next
``find_leader`` doesn't waste an RTT probing a now-stale leader before the full sweep."""

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
    fake_proto.negotiate_protocol_only = AsyncMock()
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
    fake_proto.negotiate_protocol_only = AsyncMock()
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
    """A new VOTER changes quorum size; the new majority may elect a different leader."""
    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
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
    """Promoting STANDBY → VOTER widens the voting set; the election window applies."""
    from dqlitewire import NodeRole

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
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
async def test_set_weight_preserves_last_known_leader_on_success() -> None:
    """On SUCCESS the cache stays warm — matches go-dqlite's ``Client.Weight``."""
    cluster = _make_cluster_with_cached_leader()
    cached = cluster._get_last_known_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.weight = AsyncMock()

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        await cluster.set_weight(weight=5)
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() == cached


@pytest.mark.asyncio
async def test_transfer_leadership_invalidates_cache_on_failure() -> None:
    """A leader-flip failure mid-RPC must invalidate too, not just the success path."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
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

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_add_node_invalidates_cache_on_failure() -> None:
    """add_node failure (e.g. ASSIGN fails after ADD landed) must also invalidate."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.add = AsyncMock()
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


# Read-only RPCs that also call ``find_leader`` (all but ``describe(address=...)``) must
# mirror the membership-RPC discipline: invalidate on failure so a flip doesn't strand the cache.


@pytest.mark.asyncio
async def test_cluster_info_invalidates_cache_on_failure() -> None:
    """A leader-flip mid-RPC must invalidate the cache."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    # Leadership re-confirm succeeds; the failure comes from the ``cluster()`` wire RPC.
    fake_proto.get_leader = AsyncMock(return_value=(1, "node1:9001"))
    fake_proto.cluster = AsyncMock(
        side_effect=OperationalError("not leader", code=1001, raw_message="not leader")
    )

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.cluster_info()
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_leader_info_invalidates_cache_on_failure() -> None:
    """A leader-flip mid-RPC must invalidate the cache."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.get_leader = AsyncMock(
        side_effect=OperationalError("not leader", code=1001, raw_message="not leader")
    )

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.leader_info()
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_dump_invalidates_cache_on_failure() -> None:
    """``dump`` targets the leader; a step-down mid-dump must invalidate the cache."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.dump = AsyncMock(
        side_effect=OperationalError("not leader", code=1001, raw_message="not leader")
    )

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.dump("default")
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_describe_no_address_invalidates_cache_on_failure() -> None:
    """``describe(address=None)`` targets the leader; a flip mid-RPC must invalidate."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.describe = AsyncMock(
        side_effect=OperationalError("not leader", code=1001, raw_message="not leader")
    )

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.describe()
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_describe_with_specific_address_preserves_cache_on_failure() -> None:
    """Negative twin: ``describe(address=...)`` bypasses ``find_leader``, so a per-peer
    failure must not invalidate the unrelated leader cache."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.describe = AsyncMock(
        side_effect=OperationalError("peer rejected", code=1, raw_message="peer rejected")
    )

    patches = _patch_admin(cluster, fake_proto)
    for p in patches:
        p.start()
    try:
        with pytest.raises(OperationalError):
            await cluster.describe(address="node2:9001")
    finally:
        for p in reversed(patches):
            p.stop()

    assert cluster._get_last_known_leader() == "node1:9001"
