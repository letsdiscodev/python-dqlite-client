"""Pin: ``ClusterClient.open_admin_connection`` performs version-only
negotiation (``negotiate_protocol_only``) instead of the full
``handshake()`` (version + ``ClientRequest`` + ``WelcomeResponse``).

Go-dqlite's ``client.New`` / ``NewDirectConnector.Connect`` (see
``client/client.go:56-75`` and
``internal/protocol/connector.go:316-337``) speak ONLY the 8-byte
version write on direct-admin connections — the ``ClientRequest``
registration is reserved for the leader-finding path
(``connector.go::connectAttemptOne``). Python's ``_query_leader``
correctly uses ``negotiate_protocol_only``; ``open_admin_connection``
previously did the full ``handshake()``, paying an extra RTT per
admin RPC and allocating a server-side ``g->client_id`` slot that
no admin RPC depends on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore

pytestmark = pytest.mark.asyncio


async def test_open_admin_connection_uses_version_only_negotiation() -> None:
    """Patch ``DqliteProtocol.handshake`` and ``negotiate_protocol_only``
    on the cluster module so the test can detect which one
    ``open_admin_connection`` calls. The full ``handshake`` MUST NOT
    be invoked — only ``negotiate_protocol_only`` may run.
    """
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
