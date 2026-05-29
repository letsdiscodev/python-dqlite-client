"""A handshake wire-decode ``ProtocolError`` must surface as
``DqliteConnectionError`` so the connect-retry classifier matches it; otherwise a
peer mid-restart producing a torn frame abandons instead of using the retry budget."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError, ProtocolError


@pytest.mark.asyncio
async def test_handshake_protocolerror_wrapped_as_dqlite_connection_error() -> None:
    """A ``ProtocolError`` from ``handshake()`` is rewrapped as ``DqliteConnectionError``."""
    conn = DqliteConnection("127.0.0.1:9001", database="db", timeout=2.0)

    fake_streams = (AsyncMock(spec=asyncio.StreamReader), AsyncMock(spec=asyncio.StreamWriter))
    fake_streams[1].close = lambda: None
    fake_streams[1].wait_closed = AsyncMock()

    async def stub_open(*_args: object, **_kwargs: object):
        return fake_streams

    with (
        patch("dqliteclient._dial.open_connection_with_keepalive", new=stub_open),
        patch(
            "dqliteclient.connection.DqliteProtocol.handshake",
            new=AsyncMock(side_effect=ProtocolError("simulated torn frame")),
        ),
        pytest.raises(DqliteConnectionError, match="(?i)wire decode failed"),
    ):
        await conn.connect()


@pytest.mark.asyncio
async def test_handshake_protocolerror_preserves_cause() -> None:
    """The original ProtocolError remains as ``__cause__``."""
    conn = DqliteConnection("127.0.0.1:9001", database="db", timeout=2.0)

    fake_streams = (AsyncMock(spec=asyncio.StreamReader), AsyncMock(spec=asyncio.StreamWriter))
    fake_streams[1].close = lambda: None
    fake_streams[1].wait_closed = AsyncMock()

    async def stub_open(*_args: object, **_kwargs: object):
        return fake_streams

    original = ProtocolError("simulated torn frame")
    with (
        patch("dqliteclient._dial.open_connection_with_keepalive", new=stub_open),
        patch(
            "dqliteclient.connection.DqliteProtocol.handshake",
            new=AsyncMock(side_effect=original),
        ),
    ):
        try:
            await conn.connect()
        except DqliteConnectionError as e:
            assert e.__cause__ is original
        else:
            pytest.fail("expected DqliteConnectionError")
