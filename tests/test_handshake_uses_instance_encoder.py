"""Pin: ``handshake`` and ``negotiate_protocol_only`` use the bound
``self._encoder``, not a fresh ``MessageEncoder()``. Harmless today (the version
word is invariant) but a fresh encoder would bypass per-connection caps if more
content were later routed through it."""

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
async def test_handshake_uses_instance_encoder() -> None:
    """Pin: handshake calls ``self._encoder.encode_handshake``, not a bare one."""
    proto = _make_protocol()
    sentinel = b"\x42" * 8
    proto._encoder.encode_handshake = MagicMock(return_value=sentinel)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    proto._encoder.encode_handshake.assert_called_once_with()
    payload = proto._send.call_args.args[0]
    assert payload.startswith(sentinel), (
        "handshake payload must start with self._encoder.encode_handshake() output"
    )


@pytest.mark.asyncio
async def test_negotiate_protocol_only_uses_instance_encoder() -> None:
    """negotiate_protocol_only also uses the bound encoder."""
    proto = _make_protocol()
    sentinel = b"\x99" * 8
    proto._encoder.encode_handshake = MagicMock(return_value=sentinel)
    proto._send = AsyncMock()

    await proto.negotiate_protocol_only()

    proto._encoder.encode_handshake.assert_called_once_with()
    payload = proto._send.call_args.args[0]
    assert payload == sentinel
