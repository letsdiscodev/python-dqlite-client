"""Pin: ``dial_func`` override surface for go-dqlite parity with
``WithDialFunc``.

Adds a caller-supplied dialer at every dial site:
:meth:`DqliteConnection.__init__`, :meth:`ClusterClient.__init__`
(forwarded via :meth:`ClusterClient.from_addresses` and through
:class:`ConnectionPool`), and the cancel-time
``_send_interrupt_on_fresh_socket`` helper. Default-``None`` callers
fall through to the existing :func:`open_connection_with_keepalive`
helper unchanged — this set of pins fences both the override path and
the default fall-through.

Mutual-exclusion: ``ConnectionPool(cluster=, dial_func=)`` rejects with
``ValueError`` because an externally-owned ``ClusterClient`` already
carries its own dialer; supplying both would silently desync the two.
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
    """Build a (reader, writer) pair so the dispatcher returns a
    plausible shape. Tests inspect call args; the streams themselves
    are not exercised end-to-end. ``MagicMock`` avoids asyncio's
    StreamWriter finalizer racing the test teardown when the
    transport is not a real ``WriteTransport``."""
    reader: Any = MagicMock(spec=asyncio.StreamReader)
    writer: Any = MagicMock(spec=asyncio.StreamWriter)
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return reader, writer


def test_dial_func_exported_from_top_level() -> None:
    """``DialFunc`` is part of the public surface so type-checker users
    can spell the dialer signature."""
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
    """An externally-owned ClusterClient already carries its own
    dial_func; supplying both at the pool layer is rejected to avoid
    silent divergence."""
    stub: DialFunc = AsyncMock()
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]))
    with pytest.raises(ValueError, match="dial_func cannot be combined with cluster="):
        ConnectionPool(cluster=cluster, dial_func=stub)


@pytest.mark.asyncio
async def test_create_pool_forwards_dial_func() -> None:
    """Top-level ``create_pool`` plumbs ``dial_func`` through to the
    auto-built ClusterClient."""
    stub: DialFunc = AsyncMock()
    # Stub the pool initialise so we don't need a live cluster.
    with patch.object(ConnectionPool, "initialize", new=AsyncMock()):
        pool = await create_pool(["localhost:9001"], dial_func=stub)
    assert pool._cluster._dial_func is stub


@pytest.mark.asyncio
async def test_dispatcher_calls_default_helper_when_dial_func_none() -> None:
    """``open_connection(address, dial_func=None)`` falls through to
    :func:`open_connection_with_keepalive` after parsing host/port."""
    from dqliteclient._dial import open_connection

    fake_streams = _stub_streams()
    helper = AsyncMock(return_value=fake_streams)
    with patch("dqliteclient._dial.open_connection_with_keepalive", helper):
        result = await open_connection("localhost:9001", dial_func=None)

    assert result is fake_streams
    helper.assert_awaited_once_with("localhost", 9001)


@pytest.mark.asyncio
async def test_dispatcher_uses_dial_func_with_full_address() -> None:
    """When ``dial_func`` is supplied, the dispatcher delegates the
    FULL address string opaquely (no host/port parsing)."""
    from dqliteclient._dial import open_connection

    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)
    result = await open_connection("leader.example:9001", dial_func=stub)

    assert result is fake_streams
    stub.assert_awaited_once_with("leader.example:9001")


@pytest.mark.asyncio
async def test_dispatcher_propagates_oserror_from_custom_dialer() -> None:
    """A ``dial_func`` raising :class:`OSError` propagates so the
    existing per-call-site exception arms classify it as a transient
    transport fault."""
    from dqliteclient._dial import open_connection

    stub = AsyncMock(side_effect=OSError("tls handshake failed"))
    with pytest.raises(OSError, match="tls handshake failed"):
        await open_connection("localhost:9001", dial_func=stub)


@pytest.mark.asyncio
async def test_dispatcher_propagates_timeout_error_from_custom_dialer() -> None:
    """A ``dial_func`` raising :class:`TimeoutError` propagates so the
    per-call-site arms classify as ``DqliteConnectionError(timed out)``."""
    from dqliteclient._dial import open_connection

    stub = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(TimeoutError):
        await open_connection("localhost:9001", dial_func=stub)


@pytest.mark.asyncio
async def test_dqlite_connection_connect_calls_dial_func_with_full_address() -> None:
    """End-to-end: passing ``dial_func`` to :class:`DqliteConnection``
    routes the dial through it. Patches the protocol so the test is
    dial-only — handshake/open_database/close are stubbed."""
    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    with patch("dqliteclient.connection.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
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
    """End-to-end: ``_query_leader`` dials via ``self._dial_func`` so a
    TLS-configured cluster's leader-discovery sweep also goes via TLS.
    Without this pin, a future refactor that drops the ``dial_func``
    argument at the call site would not break any test."""
    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    cluster = ClusterClient(MemoryNodeStore(["leader.example:9001"]), dial_func=stub)
    # Patch the protocol so _query_leader returns without real I/O.
    with patch("dqliteclient.cluster.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
        proto.get_leader = AsyncMock(return_value=(1, "leader.example:9001"))
        proto.close = MagicMock()
        proto_cls.return_value = proto
        await cluster._query_leader("leader.example:9001")

    stub.assert_awaited_once_with("leader.example:9001")


@pytest.mark.asyncio
async def test_open_admin_connection_uses_cluster_dial_func() -> None:
    """End-to-end: ``open_admin_connection`` dials via
    ``self._dial_func``. Symmetric pin to ``_query_leader`` so admin
    RPCs (cluster_info / dump / membership ops) also honour the
    caller's TLS configuration."""
    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    cluster = ClusterClient(MemoryNodeStore(["admin.example:9001"]), dial_func=stub)
    with patch("dqliteclient.cluster.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
        proto.close = MagicMock()
        proto_cls.return_value = proto
        async with cluster.open_admin_connection("admin.example:9001"):
            pass

    stub.assert_awaited_once_with("admin.example:9001")


@pytest.mark.asyncio
async def test_send_interrupt_on_fresh_socket_uses_dial_func() -> None:
    """The cancel-arm ``_send_interrupt_on_fresh_socket`` helper
    accepts and uses the parent connection's ``dial_func`` so a
    TLS-configured connection's INTERRUPT also goes via TLS."""
    from unittest.mock import MagicMock

    from dqliteclient.connection import _send_interrupt_on_fresh_socket

    fake_streams = _stub_streams()
    stub = AsyncMock(return_value=fake_streams)

    with patch("dqliteclient.connection.DqliteProtocol") as proto_cls:
        proto = MagicMock()
        proto.handshake = AsyncMock()
        proto._interrupt = AsyncMock()
        proto_cls.return_value = proto

        await _send_interrupt_on_fresh_socket(
            "leader.example:9001",
            db_id=42,
            dial_timeout=1.0,
            interrupt_timeout=1.0,
            dial_func=stub,
        )

    stub.assert_awaited_once_with("leader.example:9001")
