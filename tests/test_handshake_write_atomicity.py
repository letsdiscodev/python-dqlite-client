"""Pin: ``handshake`` writes the protocol-version word and the client
registration request to the wire as a SINGLE ``self._send(frame)``
call carrying a single assembled ``bytes`` payload.

The dqlite server expects the 8-byte version word followed immediately
by the ``ClientRequest`` frame. If a future refactor inserts an
``await`` between writing the version word and writing the
registration request, a ``CancelledError`` (or any suspension that
ends with the connection being recycled) could leave the server with
an incomplete handshake — the next handshake on the same socket
would be misframed.

The current code assembles both pieces into a single ``bytes`` object
before the only ``self._send`` call (see ``protocol.py``). The
``_send`` helper internally performs ``self._writer.write(frame)``
followed by ``await self._writer.drain()`` inside a single try/except
so a synchronous transport-closed RuntimeError surfaces as
``DqliteConnectionError``. These tests pin the bundling contract so
the assembly cannot drift.
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
    """Pin: the single buffer's first 8 bytes are the protocol-version
    handshake (so a regression that swaps the order is caught here)."""
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
