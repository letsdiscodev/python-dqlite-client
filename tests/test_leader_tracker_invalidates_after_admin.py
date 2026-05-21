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
async def test_set_weight_preserves_last_known_leader_on_success() -> None:
    """``set_weight`` is a read-mostly admin RPC: on SUCCESS the
    responding leader has provably just answered the RPC, so the
    leader cache stays warm. Matches go-dqlite's ``Client.Weight``
    which does not touch the leader tracker on success. Failure-
    path invalidation is exercised by the sibling test
    ``test_set_weight_per_node_does_not_invalidate_on_failure`` and
    by ``test_admin_rpcs_preserve_cache_on_success``."""
    cluster = _make_cluster_with_cached_leader()
    cached = cluster._get_last_known_leader()
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

    assert cluster._get_last_known_leader() == cached


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


# ---------------------------------------------------------------------------
# Read-only admin RPCs (cluster_info / leader_info / dump / describe) —
# the membership-mutating RPCs above already invalidate the leader cache on
# failure. The four read-only RPCs that ALSO call ``find_leader`` (every one
# except ``describe(address=<specific>)``) MUST mirror that discipline so a
# leader-flip mid-RPC doesn't leave the cache pointing at a now-stale peer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_info_invalidates_cache_on_failure() -> None:
    """``cluster_info`` is sent to the leader. A leader-flip mid-RPC
    surfaces as ``OperationalError`` / ``DqliteConnectionError`` and
    must invalidate the cache so the next ``find_leader`` runs a full
    sweep instead of paying a fast-path miss against the dead peer."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    # Re-confirm leadership round-trip succeeds; the failure surfaces
    # from the subsequent ``cluster()`` call (the wire RPC).
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
    """``leader_info`` is sent to the leader (after ``find_leader``).
    A leader-flip mid-RPC must invalidate the cache."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
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
    """``dump`` is sent to the leader (Python design choice). A
    leader step-down mid-dump is plausible given the long-lived
    socket; invalidate the cache so the next ``find_leader`` runs a
    full sweep instead of probing the dead peer."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
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
    """``describe(address=None)`` calls ``find_leader`` and sends to
    the leader. A leader-flip mid-RPC must invalidate the cache.
    ``describe(address=<specific>)`` does NOT call ``find_leader``
    and is covered by a separate negative pin below."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
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
    """Negative twin: ``describe(address=<specific>)`` bypasses
    ``find_leader`` entirely — the request lands on the named peer,
    not the leader. The leader cache is unrelated to the call and
    MUST NOT be invalidated on a per-peer failure (over-aggressive
    invalidation would force an avoidable sweep)."""
    from dqliteclient.exceptions import OperationalError

    cluster = _make_cluster_with_cached_leader()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
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

    # Cache untouched — the call did not target the leader.
    assert cluster._get_last_known_leader() == "node1:9001"
