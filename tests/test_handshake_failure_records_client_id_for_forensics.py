"""Pin: ``handshake`` records ``self._client_id`` BEFORE the wire write so a
``FailureResponse`` arm can surface the slot id. Upstream ``handle_client``
allocates the per-gateway slot before composing the response (reclaimed at TCP
close), so the id lets operators correlate the server-side trace."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import OperationalError
from dqliteclient.protocol import DqliteProtocol


@pytest.mark.asyncio
async def test_handshake_failure_message_includes_slot_breadcrumb() -> None:
    """A FailureResponse produces an OperationalError whose message carries the
    negotiated ``client_id`` for grepping server logs."""
    from dqlitewire.messages import FailureResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    failure = FailureResponse(code=42, message="server is exhausted")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=failure)),
        pytest.raises(OperationalError) as exc_info,
    ):
        await protocol.handshake(client_id=0xDEADBEEF)

    msg = str(exc_info.value)
    assert "client slot may be allocated as id=3735928559" in msg
    assert "reclaimed on TCP close" in msg


@pytest.mark.asyncio
async def test_handshake_records_client_id_before_send() -> None:
    """The negotiated id is set on the protocol BEFORE the wire write."""
    from dqlitewire.messages import WelcomeResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    seen_ids: list[int] = []

    async def fake_send(_data: bytes) -> None:
        seen_ids.append(protocol._client_id)

    welcome = WelcomeResponse(heartbeat_timeout=15000)
    with (
        patch.object(protocol, "_send", new=fake_send),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=welcome)),
    ):
        await protocol.handshake(client_id=12345)

    assert seen_ids == [12345]
    assert protocol._client_id == 12345


@pytest.mark.asyncio
async def test_handshake_failure_message_uses_random_id_when_unspecified() -> None:
    """Without a caller-supplied ``client_id``, the random id is still in the message."""
    from dqlitewire.messages import FailureResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    failure = FailureResponse(code=42, message="boom")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=failure)),
        pytest.raises(OperationalError) as exc_info,
    ):
        await protocol.handshake()

    assert protocol._client_id != 0
    assert f"id={protocol._client_id}" in str(exc_info.value)
