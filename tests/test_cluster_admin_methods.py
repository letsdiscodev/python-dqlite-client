"""``ClusterClient.cluster_info`` and ``transfer_leadership`` (mirroring go-dqlite's
``client.Cluster`` / ``client.Transfer``), both routed through ``open_admin_connection``.
Transport is mocked; live behaviour is in the integration suite."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole
from dqlitewire.messages.responses import NodeInfo

_FakeOpenConnection = Callable[[str, int], Awaitable[tuple[MagicMock, MagicMock]]]


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


def _patch_admin_connection(
    fake_proto: MagicMock,
) -> tuple[_FakeOpenConnection, MagicMock]:
    """Patch network primitives so ``open_admin_connection`` yields ``fake_proto``."""
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
async def test_cluster_info_returns_node_list_from_leader() -> None:
    """The call returns the leader's decoded NodeInfo list verbatim."""
    cluster = _make_cluster()
    nodes = [
        NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="node2:9002", role=NodeRole.VOTER),
        NodeInfo(node_id=3, address="node3:9003", role=NodeRole.VOTER),
    ]

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.cluster = AsyncMock(return_value=nodes)
    # Re-confirm leadership round-trip on the no-flip happy path.
    fake_proto.get_leader = AsyncMock(return_value=(1, "node1:9001"))

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        result = await cluster.cluster_info()

    assert result == nodes
    fake_proto.cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cluster_info_propagates_leader_unreachable_as_cluster_error() -> None:
    """``find_leader``'s ClusterError propagates unchanged — admin methods do not retry."""
    cluster = _make_cluster()

    with (
        patch.object(cluster, "find_leader", AsyncMock(side_effect=ClusterError("no leader"))),
        pytest.raises(ClusterError),
    ):
        await cluster.cluster_info()


@pytest.mark.asyncio
async def test_cluster_info_propagates_operational_error_from_leader() -> None:
    """A leader rejecting the request surfaces as OperationalError."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.cluster = AsyncMock(side_effect=OperationalError("shutting down", 1))
    fake_proto.get_leader = AsyncMock(return_value=(1, "node1:9001"))

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError),
    ):
        await cluster.cluster_info()


@pytest.mark.asyncio
async def test_transfer_leadership_sends_request_with_target_id() -> None:
    """``TransferRequest(target_node_id)`` is dispatched to the current leader."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.transfer = AsyncMock()

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.transfer_leadership(target_node_id=2)

    fake_proto.transfer.assert_awaited_once_with(2)


@pytest.mark.asyncio
async def test_transfer_leadership_rejects_non_int_target() -> None:
    """``target_node_id`` must be int; local validation raises TypeError at the call site
    rather than a cryptic wire-decode error."""
    cluster = _make_cluster()

    for bad in (True, False, "2", 2.0, None):
        with pytest.raises(TypeError, match="target_node_id must be int"):
            await cluster.transfer_leadership(target_node_id=bad)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_transfer_leadership_rejects_zero_or_negative_target() -> None:
    """Node id 0 is the upstream "no node" sentinel, so a non-positive id is never a valid
    promotion target."""
    cluster = _make_cluster()

    for bad in (0, -1, -100):
        with pytest.raises(ValueError, match="target_node_id must be >= 1"):
            await cluster.transfer_leadership(target_node_id=bad)


@pytest.mark.asyncio
async def test_transfer_leadership_propagates_leader_unreachable() -> None:
    """``find_leader`` raising ``ClusterError`` propagates unchanged."""
    cluster = _make_cluster()

    with (
        patch.object(cluster, "find_leader", AsyncMock(side_effect=ClusterError("no leader"))),
        pytest.raises(ClusterError),
    ):
        await cluster.transfer_leadership(target_node_id=2)


@pytest.mark.asyncio
async def test_transfer_leadership_propagates_server_rejection() -> None:
    """A server rejecting the transfer surfaces as OperationalError."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.transfer = AsyncMock(side_effect=OperationalError("target is not a voter", 1))

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError, match="target is not a voter"),
    ):
        await cluster.transfer_leadership(target_node_id=99)


@pytest.mark.asyncio
async def test_admin_connection_closes_writer_on_normal_exit() -> None:
    """The asynccontextmanager closes the writer on the happy path."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.cluster = AsyncMock(return_value=[])
    fake_proto.get_leader = AsyncMock(return_value=(1, "node1:9001"))

    fake_open, writer = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.cluster_info()

    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_admin_connection_closes_writer_on_protocol_error() -> None:
    """The writer must close even when the protocol method raises, or a failed transfer
    leaks a half-closed socket."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.transfer = AsyncMock(side_effect=OperationalError("rejected", 1))

    fake_open, writer = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError),
    ):
        await cluster.transfer_leadership(target_node_id=2)

    writer.close.assert_called_once()
