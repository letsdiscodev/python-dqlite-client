"""Pin: ``max_message_size`` flows from ``ClusterClient.__init__`` (and
from the ``Pool`` constructor that wraps it) into every
``DqliteProtocol`` instance that the leader-probe and admin paths
build.

A prior parity fix wired ``trust_server_heartbeat`` /
``max_total_rows`` / ``max_continuation_frames`` through the cluster
client; ``max_message_size`` was added to the pool / dbapi / SA
surface later and was not retrofitted, so operators tightening the
cap cluster-wide as a DoS hardening lever saw the cap silently
bypassed on the admin path (notably ``dump``, where a multi-GB
database arrives as one frame per file content).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_query_leader_forwards_max_message_size() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=1.0, max_message_size=1024)

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    captured: dict[str, object] = {}

    class FakeProto:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["max_message_size"] = kwargs.get("max_message_size")

        async def handshake(self) -> None:
            pass

        async def negotiate_protocol_only(self) -> None:
            pass

        async def get_leader(self) -> tuple[int, str]:
            return (1, "leader:9001")

    with (
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
        patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
    ):
        await client._query_leader("localhost:9001")

    assert captured["max_message_size"] == 1024


@pytest.mark.asyncio
async def test_open_admin_connection_forwards_max_message_size() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=1.0, max_message_size=2048)

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    captured: dict[str, object] = {}

    class FakeProto:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["max_message_size"] = kwargs.get("max_message_size")

        async def handshake(self) -> None:
            pass

    with (
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
        patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
    ):
        async with client.open_admin_connection("localhost:9001"):
            pass

    assert captured["max_message_size"] == 2048


def test_cluster_client_default_is_none() -> None:
    """When ``max_message_size`` is unset, ``None`` is stored — the
    underlying DqliteProtocol falls back to the wire-layer default
    (64 MiB)."""
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=1.0)
    assert client._max_message_size is None
