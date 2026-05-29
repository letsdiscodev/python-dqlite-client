"""Pin: a handshake ``ProtocolError`` keeps the peer-address suffix even for a
long server message. ``_failure_text`` pre-truncates before appending the suffix,
so the raise site must NOT wrap it in an outer truncator that strips the suffix."""

from __future__ import annotations

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


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


def test_handshake_protocolerror_does_not_use_outer_truncate_error() -> None:
    """Source-level pin: the handshake raise site must not wrap _failure_text in
    _truncate_error, which would strip the addr suffix."""
    import inspect

    from dqliteclient import protocol

    src = inspect.getsource(protocol.DqliteProtocol.handshake)
    assert "_truncate_error(self._failure_text" not in src, (
        "Handshake raise site must not wrap _failure_text in _truncate_error"
    )
