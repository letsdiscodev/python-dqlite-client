"""DqliteProtocol handshake: write atomicity, read timeouts, encoder use and failure diagnostics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqliteclient.protocol import _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS, DqliteProtocol
from dqlitewire.messages import FailureResponse, WelcomeResponse


def _make_proto_with_address(address: str) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = address
    return proto


def test_handshake_protocolerror_composition_preserves_addr_suffix() -> None:
    """A long FailureResponse.message still ends with the addr suffix that
    ``_failure_text`` appends, with no outer truncator stripping it."""
    proto = _make_proto_with_address("host-a:9001")
    body = "X" * 1024
    response = FailureResponse(code=1001, message=body)

    rendered = f"Handshake failed: [{response.code}] {proto._failure_text(response)}"

    err = ProtocolError(rendered)
    text = str(err)
    assert text.endswith(" to host-a:9001"), f"Addr suffix dropped from handshake error: {text!r}"
    assert "[truncated," in text


def test_handshake_protocolerror_composition_short_message_no_truncation() -> None:
    """The no-truncation path (short message) still appends the suffix verbatim."""
    proto = _make_proto_with_address("host-b:19002")
    response = FailureResponse(code=1001, message="boom")
    rendered = f"Handshake failed: [{response.code}] {proto._failure_text(response)}"
    text = str(ProtocolError(rendered))
    assert text.endswith(" to host-b:19002")
    assert "[truncated," not in text


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

    assert str(exc_info.value).startswith("Handshake failed: [42]")
    assert protocol._client_id == 0xDEADBEEF


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
        pytest.raises(OperationalError),
    ):
        await protocol.handshake()

    assert protocol._client_id != 0


def _make_proto() -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = "host-a:9001"
    return proto


def test_handshake_failure_response_operational_error_preserves_raw_message_and_code() -> None:
    """A handshake FailureResponse yields an OperationalError whose ``raw_message``
    and ``code`` match the wire response."""
    proto = _make_proto()
    body = "version negotiation failed: peer reports protocol=999"
    response = FailureResponse(code=1001, message=body)
    err = OperationalError(
        f"Handshake failed: [{response.code}] {proto._failure_text(response)}",
        response.code,
        raw_message=response.message,
    )
    assert err.raw_message == body, (
        f"handshake OperationalError must carry raw_message={body!r}; got {err.raw_message!r}"
    )
    assert err.code == 1001, f"handshake OperationalError must carry code=1001; got {err.code!r}"


def test_dqlite_connection_error_rewrap_preserves_raw_message() -> None:
    """The connect() rewrap threads ``raw_message`` from the inner OperationalError."""
    inner = OperationalError(
        "Handshake failed: [1001] long server text",
        1001,
        raw_message="long server text",
    )
    rewrap = DqliteConnectionError(
        f"Wire decode failed during handshake to host-a:9001: {inner}",
        code=getattr(inner, "code", None),
        raw_message=getattr(inner, "raw_message", None) or str(inner),
    )
    assert rewrap.raw_message == "long server text"


def _make_timeout_protocol(timeout: float, *, trust: bool) -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=timeout, trust_server_heartbeat=trust)


@pytest.mark.asyncio
async def test_default_handshake_does_not_widen_read_timeout() -> None:
    """Default ``trust_server_heartbeat=False`` does not widen the read deadline."""
    proto = _make_timeout_protocol(timeout=5.0, trust=False)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=300_000))

    await proto.handshake()

    assert proto._read_timeout == 5.0
    # Heartbeat is still recorded for diagnostics.
    assert proto._heartbeat_timeout == 300_000


@pytest.mark.asyncio
async def test_trust_handshake_widens_up_to_cap() -> None:
    """``trust=True`` with a heartbeat above the cap widens to the cap, not beyond."""
    proto = _make_timeout_protocol(timeout=5.0, trust=True)
    proto._send = AsyncMock()
    # Advertise 600 s; cap is 300.
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=600_000))

    await proto.handshake()

    assert proto._read_timeout == _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS


@pytest.mark.asyncio
async def test_trust_handshake_does_not_narrow_read_timeout() -> None:
    """A small heartbeat does not narrow the operator's larger configured timeout."""
    proto = _make_timeout_protocol(timeout=30.0, trust=True)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=1_000))

    await proto.handshake()

    assert proto._read_timeout == 30.0


@pytest.mark.asyncio
async def test_trust_handshake_zero_or_negative_heartbeat_no_widen() -> None:
    """Heartbeat <= 0 disables widening (the ``> 0`` guard keeps diagnostics accurate)."""
    proto = _make_timeout_protocol(timeout=5.0, trust=True)
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake()

    assert proto._read_timeout == 5.0


def _make_encoder_protocol() -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=5.0)


@pytest.mark.asyncio
async def test_handshake_uses_instance_encoder() -> None:
    """Pin: handshake calls ``self._encoder.encode_handshake``, not a bare one."""
    proto = _make_encoder_protocol()
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
    proto = _make_encoder_protocol()
    sentinel = b"\x99" * 8
    proto._encoder.encode_handshake = MagicMock(return_value=sentinel)
    proto._send = AsyncMock()

    await proto.negotiate_protocol_only()

    proto._encoder.encode_handshake.assert_called_once_with()
    payload = proto._send.call_args.args[0]
    assert payload == sentinel


def _make_atomic_protocol() -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=5.0)


@pytest.mark.asyncio
async def test_handshake_sends_version_and_request_in_single_call() -> None:
    """Pin: exactly one ``self._send`` invocation during handshake."""
    proto = _make_atomic_protocol()
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

    proto = _make_atomic_protocol()
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
    proto = _make_atomic_protocol()
    proto._send = AsyncMock()
    proto._read_response = AsyncMock(return_value=WelcomeResponse(heartbeat_timeout=0))

    await proto.handshake(client_id=42)

    assert proto._send.call_count == 1
