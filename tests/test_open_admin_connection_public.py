"""``ClusterClient.open_admin_connection`` is the public direct-to-
node primitive — analogue of go-dqlite's
``NewDirectConnector(id, address, options...).Connect(ctx)``
(``client.go:358-367``).

Pre-rename it was the leading-underscore private ``_admin_connection``;
external callers building bespoke admin tooling (talk-to-this-specific-
node) had no documented entrypoint. The rename is forward-only — no
external imports of the private name existed in the tree.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_open_admin_connection_is_public_method() -> None:
    """The method is exposed without the leading underscore. The
    private name no longer exists — pin against accidentally
    re-introducing the underscore."""
    assert hasattr(ClusterClient, "open_admin_connection")
    assert not hasattr(ClusterClient, "_admin_connection")


@pytest.mark.asyncio
async def test_open_admin_connection_yields_handshaken_protocol() -> None:
    """End-to-end: opening the public method against a mocked
    transport yields a protocol with handshake completed and the
    socket closed on exit."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    with (
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        async with cluster.open_admin_connection("localhost:9001") as proto:
            assert proto is fake_proto

    fake_proto.handshake.assert_awaited_once()
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_open_admin_connection_closes_writer_on_exception() -> None:
    """A caller-raised exception inside the ``async with`` body must
    still trigger socket cleanup — same discipline as the existing
    private form."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    with (  # noqa: SIM117
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        with pytest.raises(RuntimeError, match="caller error"):
            async with cluster.open_admin_connection("localhost:9001"):
                raise RuntimeError("caller error")

    writer.close.assert_called_once()
