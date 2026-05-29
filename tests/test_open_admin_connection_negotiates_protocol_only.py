"""``open_admin_connection`` does version-only negotiation, not the full ``handshake()``:
the full handshake costs an extra RTT and allocates a server-side g->client_id slot no
admin RPC needs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore

pytestmark = pytest.mark.asyncio


async def test_open_admin_connection_uses_version_only_negotiation() -> None:
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]))

    fake_reader = MagicMock()
    fake_writer = MagicMock()
    fake_writer.close = MagicMock()
    fake_writer.wait_closed = AsyncMock()

    async def fake_open(_address: str, *, dial_func: object = None) -> object:
        return fake_reader, fake_writer

    handshake_called = False
    negotiate_called = False

    async def fake_handshake(self: object, client_id: int | None = None) -> int:
        nonlocal handshake_called
        handshake_called = True
        return 15000

    async def fake_negotiate(self: object) -> None:
        nonlocal negotiate_called
        negotiate_called = True

    with (
        patch("dqliteclient.cluster.open_connection", side_effect=fake_open),
        patch.object(
            __import__("dqliteclient.protocol", fromlist=["DqliteProtocol"]).DqliteProtocol,
            "handshake",
            fake_handshake,
        ),
        patch.object(
            __import__("dqliteclient.protocol", fromlist=["DqliteProtocol"]).DqliteProtocol,
            "negotiate_protocol_only",
            fake_negotiate,
        ),
    ):
        async with cluster.open_admin_connection("localhost:9001"):
            pass

    assert negotiate_called, (
        "open_admin_connection must call DqliteProtocol.negotiate_protocol_only"
    )
    assert not handshake_called, (
        "open_admin_connection must NOT call the full DqliteProtocol.handshake "
        "(it allocates a server-side g->client_id slot no admin RPC needs and "
        "costs one extra RTT per admin call)"
    )
