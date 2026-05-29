"""add_node partial-failure: ADD lands but ASSIGN raises; recovery is to re-run assign_role."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole

_FakeOpenConnection = Callable[[str, int], Awaitable[tuple[MagicMock, MagicMock]]]


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


def _patch_admin_connection() -> tuple[_FakeOpenConnection, MagicMock]:
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open_connection(
        host: str, port: int, **_kwargs: object
    ) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    return fake_open_connection, writer


@pytest.mark.asyncio
async def test_add_node_assign_failure_propagates_and_invalidates_leader_cache() -> None:
    """ASSIGN failure propagates and the finally invalidates the leader cache."""
    cluster = _make_cluster()

    cluster._set_last_known_leader("node1:9001")
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.add = AsyncMock()
    fake_proto.assign = AsyncMock(
        side_effect=OperationalError("transient leader-flip on assign", 1)
    )
    fake_open, _ = _patch_admin_connection()

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError, match="transient leader-flip on assign"),
    ):
        await cluster.add_node(node_id=42, address="node42:9001", role=NodeRole.VOTER)

    fake_proto.add.assert_awaited_once_with(42, "node42:9001")
    fake_proto.assign.assert_awaited_once_with(42, NodeRole.VOTER)

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_add_node_partial_failure_recovery_via_assign_role_converges() -> None:
    """After ADD-then-ASSIGN partial failure, a follow-up assign_role converges."""
    cluster = _make_cluster()

    # Phase 1: add_node with ADD-success / ASSIGN-failure.
    fake_proto_phase1 = MagicMock()
    fake_proto_phase1.handshake = AsyncMock()
    fake_proto_phase1.negotiate_protocol_only = AsyncMock()
    fake_proto_phase1.add = AsyncMock()
    fake_proto_phase1.assign = AsyncMock(side_effect=OperationalError("assign failed", 1))
    fake_open_phase1, _ = _patch_admin_connection()

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open_phase1),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto_phase1),
        pytest.raises(OperationalError, match="assign failed"),
    ):
        await cluster.add_node(node_id=99, address="node99:9001", role=NodeRole.VOTER)

    fake_proto_phase1.add.assert_awaited_once_with(99, "node99:9001")
    fake_proto_phase1.assign.assert_awaited_once_with(99, NodeRole.VOTER)

    # Phase 2: caller runs assign_role to converge.
    fake_proto_phase2 = MagicMock()
    fake_proto_phase2.handshake = AsyncMock()
    fake_proto_phase2.negotiate_protocol_only = AsyncMock()
    fake_proto_phase2.assign = AsyncMock()
    fake_open_phase2, _ = _patch_admin_connection()

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open_phase2),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto_phase2),
    ):
        await cluster.assign_role(node_id=99, role=NodeRole.VOTER)

    fake_proto_phase2.assign.assert_awaited_once_with(99, NodeRole.VOTER)
