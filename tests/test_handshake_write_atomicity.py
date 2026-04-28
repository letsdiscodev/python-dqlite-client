"""Pin: ``handshake`` writes the protocol-version word and the client
registration request to the wire as a SINGLE ``writer.write(...)`` call,
followed by a SINGLE ``await self._send()`` (drain).

The dqlite server expects the 8-byte version word followed immediately
by the ``ClientRequest`` frame. If a future refactor inserts an
``await`` between writing the version word and writing the
registration request, a ``CancelledError`` (or any suspension that
ends with the connection being recycled) could leave the server with
an incomplete handshake — the next handshake on the same socket
would be misframed.

The current code assembles both pieces into a single ``bytes`` object
before the only ``writer.write`` call (see ``protocol.py``). These
tests pin that contract so the assembly cannot drift.
"""

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
async def test_handshake_writes_version_and_request_in_single_write() -> None:
    """Pin: exactly one ``writer.write`` invocation during handshake."""
    proto = _make_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    write_mock = proto._writer.write
    assert write_mock.call_count == 1, (  # type: ignore[attr-defined]
        "handshake must encode version word + ClientRequest as a single "
        "buffer and call writer.write once; multiple writes risk leaving "
        "the server with a torn handshake on cancellation"
    )


@pytest.mark.asyncio
async def test_handshake_single_write_payload_starts_with_version_word() -> None:
    """Pin: the single buffer's first 8 bytes are the protocol-version
    handshake (so a regression that swaps the order is caught here)."""
    from dqlitewire.codec import MessageEncoder

    proto = _make_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    payload = proto._writer.write.call_args.args[0]  # type: ignore[attr-defined]
    expected_prefix = MessageEncoder().encode_handshake()
    assert payload.startswith(expected_prefix), (
        "handshake payload must begin with the 8-byte version word"
    )
    assert len(payload) > len(expected_prefix), (
        "handshake payload must include the ClientRequest frame after the version word"
    )


@pytest.mark.asyncio
async def test_handshake_drain_called_once() -> None:
    """Pin: exactly one drain on the assembled buffer (no per-piece flush)."""
    proto = _make_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    assert proto._send.call_count == 1
