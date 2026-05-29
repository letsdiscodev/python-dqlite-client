"""Pin: ``handshake`` writes the version word + ClientRequest as a SINGLE
``self._send`` call with one assembled ``bytes`` payload. An ``await`` between the
two writes could let a cancellation leave the server with a misframed handshake."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import WelcomeResponse


def _make_protocol() -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=5.0)


@pytest.mark.asyncio
async def test_handshake_sends_version_and_request_in_single_call() -> None:
    """Pin: exactly one ``self._send`` invocation during handshake."""
    proto = _make_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    assert proto._send.call_count == 1, (
        "handshake must assemble version word + ClientRequest as a single "
        "buffer and call _send once; multiple sends risk leaving the "
        "server with a torn handshake on cancellation"
    )


@pytest.mark.asyncio
async def test_handshake_single_send_payload_starts_with_version_word() -> None:
    """Pin: the buffer's first 8 bytes are the version word (catches an order swap)."""
    from dqlitewire.codec import MessageEncoder

    proto = _make_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    payload = proto._send.call_args.args[0]
    expected_prefix = MessageEncoder().encode_handshake()
    assert payload.startswith(expected_prefix), (
        "handshake payload must begin with the 8-byte version word"
    )
    assert len(payload) > len(expected_prefix), (
        "handshake payload must include the ClientRequest frame after the version word"
    )


@pytest.mark.asyncio
async def test_handshake_send_called_once() -> None:
    """Pin: exactly one drain on the assembled buffer (no per-piece flush)."""
    proto = _make_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    assert proto._send.call_count == 1
