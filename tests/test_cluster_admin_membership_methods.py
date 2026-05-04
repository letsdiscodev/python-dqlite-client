"""Cluster-layer tests for the new admin methods on ``ClusterClient``.

Covers ``leader_info``, ``add_node``, ``assign_role``, ``remove_node``,
``describe``, ``set_weight``, ``dump`` — the methods added to mirror
go-dqlite's full ``Client`` admin surface.

Sister of ``test_cluster_admin_methods.py`` (which covers the
earlier-added ``cluster_info`` and ``transfer_leadership``). Mocks
the transport layer so each method's:
- input validation,
- leader-routing (admin call goes to ``find_leader()``'s result),
- protocol-method dispatch,
- error propagation,
- cleanup discipline,

is covered without a live cluster. Live-cluster behaviour is in the
integration suite.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, LeaderInfo, NodeMetadata
from dqliteclient.exceptions import ClusterError, OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole

_FakeOpenConnection = Callable[[str, int], Awaitable[tuple[MagicMock, MagicMock]]]


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


def _patch_admin_connection(
    fake_proto: MagicMock,
) -> tuple[_FakeOpenConnection, MagicMock]:
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open_connection(host: str, port: int) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    return fake_open_connection, writer


# --- leader_info ---


@pytest.mark.asyncio
async def test_leader_info_returns_leader_info_dataclass() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.get_leader = AsyncMock(return_value=(7, "node7:9001"))
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node7:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        result = await cluster.leader_info()

    assert result == LeaderInfo(node_id=7, address="node7:9001")


@pytest.mark.asyncio
async def test_leader_info_returns_none_for_no_leader_known() -> None:
    """Server's legitimate ``(0, "")`` reply during a re-election
    surfaces as ``None`` — never confabulated as a real leader."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.get_leader = AsyncMock(return_value=(0, ""))
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        result = await cluster.leader_info()

    assert result is None


@pytest.mark.asyncio
async def test_leader_info_propagates_cluster_error() -> None:
    cluster = _make_cluster()
    with (
        patch.object(cluster, "find_leader", AsyncMock(side_effect=ClusterError("no leader"))),
        pytest.raises(ClusterError),
    ):
        await cluster.leader_info()


# --- add_node ---


@pytest.mark.asyncio
async def test_add_node_default_role_spare_no_assign_followup() -> None:
    """The default ``role=NodeRole.SPARE`` matches the underlying ADD
    op's implicit landing role; no follow-up ``assign`` should run."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.add = AsyncMock()
    fake_proto.assign = AsyncMock()
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.add_node(node_id=42, address="node42:9001")

    fake_proto.add.assert_awaited_once_with(42, "node42:9001")
    fake_proto.assign.assert_not_called()


@pytest.mark.asyncio
async def test_add_node_voter_role_runs_assign_followup() -> None:
    """``role=NodeRole.VOTER`` triggers the second-phase ASSIGN —
    matches go-dqlite's ``Client.Add`` two-phase semantic."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.add = AsyncMock()
    fake_proto.assign = AsyncMock()
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.add_node(node_id=42, address="node42:9001", role=NodeRole.VOTER)

    fake_proto.add.assert_awaited_once_with(42, "node42:9001")
    fake_proto.assign.assert_awaited_once_with(42, NodeRole.VOTER)


@pytest.mark.asyncio
async def test_add_node_rejects_invalid_node_id() -> None:
    cluster = _make_cluster()
    for bad in (True, False, "2", 2.0, None):
        with pytest.raises(TypeError, match="node_id must be int"):
            await cluster.add_node(node_id=bad, address="x:1")  # type: ignore[arg-type]
    for bad in (0, -1):
        with pytest.raises(ValueError, match="node_id must be >= 1"):
            await cluster.add_node(node_id=bad, address="x:1")


@pytest.mark.asyncio
async def test_add_node_rejects_invalid_address() -> None:
    cluster = _make_cluster()
    with pytest.raises(TypeError, match="address must be a non-empty str"):
        await cluster.add_node(node_id=1, address="")
    with pytest.raises(TypeError, match="address must be a non-empty str"):
        await cluster.add_node(node_id=1, address=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_add_node_rejects_invalid_role() -> None:
    cluster = _make_cluster()
    with pytest.raises(TypeError, match="role must be a NodeRole"):
        await cluster.add_node(node_id=1, address="x:1", role=0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_add_node_propagates_server_rejection() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.add = AsyncMock(side_effect=OperationalError(1, "id already in cluster"))
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError, match="already in cluster"),
    ):
        await cluster.add_node(node_id=1, address="x:1")


# --- assign_role ---


@pytest.mark.asyncio
async def test_assign_role_dispatches_with_args() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.assign = AsyncMock()
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.assign_role(node_id=2, role=NodeRole.STANDBY)

    fake_proto.assign.assert_awaited_once_with(2, NodeRole.STANDBY)


@pytest.mark.asyncio
async def test_assign_role_rejects_invalid_inputs() -> None:
    cluster = _make_cluster()
    with pytest.raises(TypeError, match="node_id must be int"):
        await cluster.assign_role(node_id="2", role=NodeRole.VOTER)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="node_id must be >= 1"):
        await cluster.assign_role(node_id=0, role=NodeRole.VOTER)
    with pytest.raises(TypeError, match="role must be a NodeRole"):
        await cluster.assign_role(node_id=1, role=0)  # type: ignore[arg-type]


# --- remove_node ---


@pytest.mark.asyncio
async def test_remove_node_dispatches_with_node_id() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.remove = AsyncMock()
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.remove_node(node_id=3)

    fake_proto.remove.assert_awaited_once_with(3)


@pytest.mark.asyncio
async def test_remove_node_rejects_invalid_inputs() -> None:
    cluster = _make_cluster()
    with pytest.raises(TypeError):
        await cluster.remove_node(node_id="3")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await cluster.remove_node(node_id=0)


# --- describe ---


@pytest.mark.asyncio
async def test_describe_returns_node_metadata_dataclass() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_response = MagicMock()
    fake_response.failure_domain = 42
    fake_response.weight = 7
    fake_proto.describe = AsyncMock(return_value=fake_response)
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        result = await cluster.describe()

    assert result == NodeMetadata(failure_domain=42, weight=7)


@pytest.mark.asyncio
async def test_describe_explicit_address_skips_leader_lookup() -> None:
    """``describe(address=...)`` targets a specific node without
    going through ``find_leader`` — matches the per-node nature of
    go-dqlite's ``Describe``."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_response = MagicMock()
    fake_response.failure_domain = 1
    fake_response.weight = 2
    fake_proto.describe = AsyncMock(return_value=fake_response)
    fake_open, _ = _patch_admin_connection(fake_proto)

    find_leader_mock = AsyncMock(return_value="node1:9001")

    with (
        patch.object(cluster, "find_leader", find_leader_mock),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.describe(address="node3:9003")

    find_leader_mock.assert_not_called()


# --- set_weight ---


@pytest.mark.asyncio
async def test_set_weight_dispatches_with_value() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.weight = AsyncMock()
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.set_weight(weight=42)

    fake_proto.weight.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_set_weight_explicit_address_skips_leader_lookup() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.weight = AsyncMock()
    fake_open, _ = _patch_admin_connection(fake_proto)

    find_leader_mock = AsyncMock(return_value="node1:9001")

    with (
        patch.object(cluster, "find_leader", find_leader_mock),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.set_weight(weight=10, address="node2:9002")

    find_leader_mock.assert_not_called()


@pytest.mark.asyncio
async def test_set_weight_rejects_invalid_inputs() -> None:
    cluster = _make_cluster()
    for bad in (True, "5", 5.0, None):
        with pytest.raises(TypeError, match="weight must be int"):
            await cluster.set_weight(weight=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight must be >= 0"):
        await cluster.set_weight(weight=-1)


# --- dump ---


@pytest.mark.asyncio
async def test_dump_returns_files_dict() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    files = {"main": b"x" * 4096, "main-wal": b"y" * 8}
    fake_proto.dump = AsyncMock(return_value=files)
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        result = await cluster.dump(database="main")

    assert result == files
    fake_proto.dump.assert_awaited_once_with("main")


@pytest.mark.asyncio
async def test_dump_default_database_name() -> None:
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.dump = AsyncMock(return_value={})
    fake_open, _ = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.dump()

    fake_proto.dump.assert_awaited_once_with("default")


@pytest.mark.asyncio
async def test_dump_rejects_invalid_database_name() -> None:
    cluster = _make_cluster()
    with pytest.raises(TypeError, match="database must be a non-empty str"):
        await cluster.dump(database="")
    with pytest.raises(TypeError):
        await cluster.dump(database=None)  # type: ignore[arg-type]


# --- open_admin_connection cleanup discipline (sanity for new methods) ---


@pytest.mark.asyncio
async def test_describe_closes_writer_on_exit() -> None:
    """The asynccontextmanager closes the writer on the happy path
    for new methods too — sanity that ``open_admin_connection``
    discipline applies to the new admin surface."""
    cluster = _make_cluster()

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_response = MagicMock()
    fake_response.failure_domain = 1
    fake_response.weight = 2
    fake_proto.describe = AsyncMock(return_value=fake_response)
    fake_open, writer = _patch_admin_connection(fake_proto)

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        await cluster.describe()

    writer.close.assert_called_once()
