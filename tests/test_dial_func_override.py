"""Pin: ``dial_func`` override surface (go-dqlite ``WithDialFunc`` parity)
at every dial site, plus the default-``None`` fall-through to
``open_connection_with_keepalive``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import dqliteclient
from dqliteclient import (
    ClusterClient,
    ConnectionPool,
    DialFunc,
    DqliteConnection,
    MemoryNodeStore,
    create_pool,
)


def _stub_streams() -> tuple[Any, Any]:
    """MagicMock (reader, writer); avoids the StreamWriter finalizer
    racing teardown when the transport is not a real WriteTransport."""
    reader: Any = MagicMock(spec=asyncio.StreamReader)
    writer: Any = MagicMock(spec=asyncio.StreamWriter)
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return reader, writer


def test_dial_func_exported_from_top_level() -> None:
    assert "DialFunc" in dqliteclient.__all__
    assert dqliteclient.DialFunc is DialFunc


def test_dqlite_connection_stores_dial_func_on_instance() -> None:
    stub: DialFunc = AsyncMock()
    conn = DqliteConnection("localhost:9001", dial_func=stub)
    assert conn._dial_func is stub


def test_dqlite_connection_default_is_none() -> None:
    conn = DqliteConnection("localhost:9001")
    assert conn._dial_func is None


def test_cluster_client_stores_dial_func_on_instance() -> None:
    stub: DialFunc = AsyncMock()
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]), dial_func=stub)
    assert cluster._dial_func is stub


def test_cluster_from_addresses_forwards_dial_func() -> None:
    stub: DialFunc = AsyncMock()
    cluster = ClusterClient.from_addresses(["localhost:9001"], dial_func=stub)
    assert cluster._dial_func is stub


def test_pool_node_store_branch_forwards_dial_func() -> None:
    stub: DialFunc = AsyncMock()
    pool = ConnectionPool(node_store=MemoryNodeStore(["localhost:9001"]), dial_func=stub)
    assert pool._cluster._dial_func is stub


def test_pool_addresses_branch_forwards_dial_func() -> None:
    stub: DialFunc = AsyncMock()
    pool = ConnectionPool(addresses=["localhost:9001"], dial_func=stub)
    assert pool._cluster._dial_func is stub


def test_pool_cluster_and_dial_func_mutex() -> None:
    # An externally-owned ClusterClient carries its own dial_func;
    # supplying both at the pool layer would silently desync them.
    stub: DialFunc = AsyncMock()
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]))
    with pytest.raises(ValueError, match="dial_func cannot be combined with cluster="):
        ConnectionPool(cluster=cluster, dial_func=stub)


@pytest.mark.asyncio
async def test_create_pool_forwards_dial_func() -> None:
    stub: DialFunc = AsyncMock()
    with patch.object(ConnectionPool, "initialize", new=AsyncMock()):
        pool = await create_pool(["localhost:9001"], dial_func=stub)
    assert pool._cluster._dial_func is stub


@pytest.mark.asyncio
async def test_dispatcher_calls_default_helper_when_dial_func_none() -> None:
    """dial_func=None falls through to open_connection_with_keepalive
    after parsing host/port."""
    from dqliteclient._dial import open_connection

    fake_streams = _stub_streams()
    helper = AsyncMock(return_value=fake_streams)
    with patch("dqliteclient._dial.open_connection_with_keepalive", helper):
        result = await open_connection("localhost:9001", dial_func=None)

    assert result is fake_streams
    helper.assert_awaited_once_with("localhost", 9001)


@pytest.mark.asyncio
async def test_dispatcher_uses_dial_func_with_full_address() -> None:
    """A supplied dial_func receives the FULL address opaquely (no parsing)."""
    from dqliteclient._dial import open_connection

    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)
    result = await open_connection("leader.example:9001", dial_func=stub)

    assert result is fake_streams
    stub.assert_awaited_once_with("leader.example:9001")


@pytest.mark.asyncio
async def test_dispatcher_propagates_oserror_from_custom_dialer() -> None:
    """OSError from dial_func propagates so per-call-site arms classify
    it as a transient transport fault."""
    from dqliteclient._dial import open_connection

    stub = AsyncMock(side_effect=OSError("tls handshake failed"))
    with pytest.raises(OSError, match="tls handshake failed"):
        await open_connection("localhost:9001", dial_func=stub)


@pytest.mark.asyncio
async def test_dispatcher_propagates_timeout_error_from_custom_dialer() -> None:
    """TimeoutError from dial_func propagates so arms classify as
    DqliteConnectionError(timed out)."""
    from dqliteclient._dial import open_connection

    stub = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(TimeoutError):
        await open_connection("localhost:9001", dial_func=stub)


@pytest.mark.asyncio
async def test_dqlite_connection_connect_calls_dial_func_with_full_address() -> None:
    """dial_func passed to DqliteConnection routes the dial through it."""
    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    with patch("dqliteclient.connection.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
        proto.negotiate_protocol_only = AsyncMock()
        proto.open_database = AsyncMock(return_value=42)
        proto.close = MagicMock()
        proto._writer = fake_streams[1]
        proto._client_id = 1
        proto_cls.return_value = proto

        conn = DqliteConnection(
            "leader.example:9001",
            database="db",
            dial_func=stub,
        )
        try:
            await conn.connect()
        finally:
            await conn.close()

    stub.assert_awaited_once_with("leader.example:9001")


@pytest.mark.asyncio
async def test_query_leader_uses_cluster_dial_func() -> None:
    """_query_leader dials via self._dial_func so a TLS cluster's
    leader-discovery sweep also goes via TLS."""
    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    cluster = ClusterClient(MemoryNodeStore(["leader.example:9001"]), dial_func=stub)
    with patch("dqliteclient.cluster.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
        proto.negotiate_protocol_only = AsyncMock()
        proto.negotiate_protocol_only = AsyncMock()
        proto.get_leader = AsyncMock(return_value=(1, "leader.example:9001"))
        proto.close = MagicMock()
        proto_cls.return_value = proto
        await cluster._query_leader("leader.example:9001")

    stub.assert_awaited_once_with("leader.example:9001")


@pytest.mark.asyncio
async def test_open_admin_connection_uses_cluster_dial_func() -> None:
    """open_admin_connection dials via self._dial_func so admin RPCs
    also honour the caller's TLS configuration."""
    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    cluster = ClusterClient(MemoryNodeStore(["admin.example:9001"]), dial_func=stub)
    with patch("dqliteclient.cluster.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
        proto.negotiate_protocol_only = AsyncMock()
        proto.close = MagicMock()
        proto_cls.return_value = proto
        async with cluster.open_admin_connection("admin.example:9001"):
            pass

    stub.assert_awaited_once_with("admin.example:9001")
