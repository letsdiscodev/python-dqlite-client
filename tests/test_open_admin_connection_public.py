"""``ClusterClient.open_admin_connection`` is the public direct-to-node primitive
(renamed from the private ``_admin_connection``)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_open_admin_connection_is_public_method() -> None:
    assert hasattr(ClusterClient, "open_admin_connection")
    assert not hasattr(ClusterClient, "_admin_connection")


@pytest.mark.asyncio
async def test_open_admin_connection_yields_handshaken_protocol() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    with (
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        async with cluster.open_admin_connection("localhost:9001") as proto:
            assert proto is fake_proto

    fake_proto.negotiate_protocol_only.assert_awaited_once()
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_open_admin_connection_closes_writer_on_exception() -> None:
    """A caller-raised exception inside the ``async with`` body still triggers socket cleanup."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    with (  # noqa: SIM117
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
    ):
        with pytest.raises(RuntimeError, match="caller error"):
            async with cluster.open_admin_connection("localhost:9001"):
                raise RuntimeError("caller error")

    writer.close.assert_called_once()
