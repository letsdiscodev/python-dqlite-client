"""Pin: handshake-time ``ProtocolError`` raised on a ``FailureResponse``
preserves the peer-address attribution suffix even when the server
message is long.

``DqliteProtocol._failure_text`` pre-truncates the body before
appending the addr suffix; the handshake raise site MUST NOT wrap the
result in an outer truncator that would re-truncate over the suffix
and silently strip the peer attribution. The most-likely-paged class
of handshake error (a flaky / mis-configured peer) is exactly the one
that produces a long FailureResponse body — losing the suffix exactly
when an operator needs to know "which node?" defeats the purpose of
the suffix.
"""

from __future__ import annotations

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


def _make_proto_with_address(address: str) -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = address
    return proto


def test_handshake_protocolerror_composition_preserves_addr_suffix() -> None:
    """The production handshake raise site composes:

        ProtocolError(f"Handshake failed: [{code}] {self._failure_text(response)}")

    With a long FailureResponse.message, the result must end with the
    addr suffix that ``_failure_text`` appends — i.e. NO outer
    truncator can wrap _failure_text and strip the suffix.
    """
    proto = _make_proto_with_address("host-a:9001")
    body = "X" * 1024
    response = FailureResponse(code=1001, message=body)

    rendered = f"Handshake failed: [{response.code}] {proto._failure_text(response)}"

    # The ProtocolError text the operator sees must end with the
    # addr suffix — that is the load-bearing peer attribution.
    err = ProtocolError(rendered)
    text = str(err)
    assert text.endswith(" to host-a:9001"), f"Addr suffix dropped from handshake error: {text!r}"
    # Defensive cap on the body still applies.
    assert "[truncated," in text


def test_handshake_protocolerror_composition_short_message_no_truncation() -> None:
    """Short messages do not need truncation; the suffix is still
    appended verbatim. Pin that the no-truncation path also keeps
    the suffix (defensive against an over-eager outer truncator
    being re-introduced)."""
    proto = _make_proto_with_address("host-b:19002")
    response = FailureResponse(code=1001, message="boom")
    rendered = f"Handshake failed: [{response.code}] {proto._failure_text(response)}"
    text = str(ProtocolError(rendered))
    assert text.endswith(" to host-b:19002")
    assert "[truncated," not in text


def test_handshake_protocolerror_does_not_use_outer_truncate_error() -> None:
    """Source-level pin: the handshake raise site at protocol.py
    must NOT wrap _failure_text in _truncate_error — that wrapping
    silently strips the addr suffix the inner pre-truncation just
    preserved. This guard catches a regression that would re-introduce
    the outer wrap.
    """
    import inspect

    from dqliteclient import protocol

    # Locate the _handshake method and read its source.
    src = inspect.getsource(protocol.DqliteProtocol.handshake)
    # The handshake's FailureResponse arm must not call
    # _truncate_error on top of self._failure_text.
    assert "_truncate_error(self._failure_text" not in src, (
        "Handshake raise site must not wrap _failure_text in _truncate_error"
    )
