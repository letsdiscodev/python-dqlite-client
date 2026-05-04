"""``ClusterClient.cluster_info`` and ``transfer_leadership``.

Mirrors the spec-level admin operations ``go-dqlite/client.Cluster``
and ``go-dqlite/client.Transfer``. Both methods route through
:meth:`ClusterClient.open_admin_connection`, which opens a one-shot
connection to the current leader, runs a single round-trip, and
tears the socket down with a bounded shutdown drain.

Tests at this layer mock the transport so the protocol-level methods
can be exercised without a live cluster. Live-cluster behaviour is
covered by the integration suite when the test cluster is
available.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole
from dqlitewire.messages.responses import NodeInfo

# Shape of ``open_connection_with_keepalive`` for the fake we
# substitute in ``_patch_admin_connection``.
_FakeOpenConnection = Callable[[str, int], Awaitable[tuple[MagicMock, MagicMock]]]


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


def _patch_admin_connection(
    fake_proto: MagicMock,
) -> tuple[_FakeOpenConnection, MagicMock]:
    """Patch the network primitives so ``open_admin_connection``
    yields ``fake_proto`` without touching a real socket."""
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open_connection(
        host: str, port: int, **_kwargs: object
    ) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    return fake_open_connection, writer


# --- cluster_info ---


@pytest.mark.asyncio
async def test_cluster_info_returns_node_list_from_leader() -> None:
    """Healthy path: leader replies with three voters; the call
    returns the decoded :class:`NodeInfo` list verbatim."""
    cluster = _make_cluster()
    nodes = [
        NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="node2:9002", role=NodeRole.VOTER),
        NodeInfo(node_id=3, address="node3:9003", role=NodeRole.VOTER),
    ]

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.cluster = AsyncMock(return_value=nodes)

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
    """``find_leader`` raising ``ClusterError`` propagates unchanged
    — admin methods do not retry; the caller decides."""
    cluster = _make_cluster()

    with (
        patch.object(cluster, "find_leader", AsyncMock(side_effect=ClusterError("no leader"))),
        pytest.raises(ClusterError),
    ):
        await cluster.cluster_info()


@pytest.mark.asyncio
async def test_cluster_info_propagates_operational_error_from_leader() -> None:
    """A leader rejecting the request (e.g. mid-shutdown) surfaces as
    ``OperationalError`` — same shape as any other dqlite failure
    response."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.cluster = AsyncMock(side_effect=OperationalError(1, "shutting down"))

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError),
    ):
        await cluster.cluster_info()


# --- transfer_leadership ---


@pytest.mark.asyncio
async def test_transfer_leadership_sends_request_with_target_id() -> None:
    """Healthy path: ``TransferRequest(target_node_id)`` is dispatched
    to the current leader and the call returns ``None``."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.transfer = AsyncMock()

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        # ``transfer_leadership`` is typed as -> None; the assertion
        # covered by mypy is ``no return value``. Call without binding,
        # then verify the wire dispatch happened.
        await cluster.transfer_leadership(target_node_id=2)

    fake_proto.transfer.assert_awaited_once_with(2)


@pytest.mark.asyncio
async def test_transfer_leadership_rejects_non_int_target() -> None:
    """``target_node_id`` must be ``int`` (not ``bool``, not ``str``,
    not ``float``). Local validation surfaces ``TypeError`` at the
    call site rather than a cryptic wire-decode error from the
    server."""
    cluster = _make_cluster()

    for bad in (True, False, "2", 2.0, None):
        with pytest.raises(TypeError, match="target_node_id must be int"):
            await cluster.transfer_leadership(target_node_id=bad)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_transfer_leadership_rejects_zero_or_negative_target() -> None:
    """Node id 0 is the upstream "no node" sentinel
    (``LeaderResponse.node_id == 0`` means "no leader known"); a
    non-positive id is never a valid promotion target."""
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
    """A server rejecting the transfer (target not a voter, target
    unreachable, cluster mid-flux) surfaces as ``OperationalError``."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.transfer = AsyncMock(side_effect=OperationalError(1, "target is not a voter"))

    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError, match="target is not a voter"),
    ):
        await cluster.transfer_leadership(target_node_id=99)


# --- open_admin_connection cleanup discipline ---


@pytest.mark.asyncio
async def test_admin_connection_closes_writer_on_normal_exit() -> None:
    """Sanity: the asynccontextmanager closes the writer on the
    happy path. Mirrors the discipline in ``_query_leader``."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.cluster = AsyncMock(return_value=[])

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
    """The writer must close even when the protocol method raises —
    otherwise a failed transfer would leak a half-closed socket."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.transfer = AsyncMock(side_effect=OperationalError(1, "rejected"))

    fake_open, writer = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError),
    ):
        await cluster.transfer_leadership(target_node_id=2)

    writer.close.assert_called_once()
