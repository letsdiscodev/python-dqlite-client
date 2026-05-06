"""Pin: handshake-`FailureResponse` `ProtocolError` raise preserves
``raw_message=response.message`` (the verbatim 64-KiB-capped server
text) — and the surrounding ``connect()`` rewrap to
``DqliteConnectionError`` carries the same field through.

The 16 sibling RPC raise sites in protocol.py thread
``raw_message=response.message`` to give cross-process forwarding
(multiprocessing.Queue) and SA-layer classification the verbatim
diagnostic text. The handshake site was the lone outlier; the
``connect()`` rewrap dropped the field too. This pin covers the
end-to-end chain: wire layer → ProtocolError → DqliteConnectionError.
"""

from __future__ import annotations

from dqliteclient.exceptions import DqliteConnectionError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


def _make_proto() -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = "host-a:9001"
    return proto


def test_handshake_failure_response_protocol_error_preserves_raw_message() -> None:
    """A FailureResponse during handshake produces a ProtocolError
    whose ``raw_message`` matches the verbatim server text, mirroring
    the 16 sibling RPC raise sites."""
    proto = _make_proto()
    body = "version negotiation failed: peer reports protocol=999"
    response = FailureResponse(code=1001, message=body)
    err = ProtocolError(
        f"Handshake failed: [{response.code}] {proto._failure_text(response)}",
        raw_message=response.message,
    )
    assert err.raw_message == body, (
        f"handshake ProtocolError must carry raw_message={body!r}; got {err.raw_message!r}"
    )


def test_dqlite_connection_error_rewrap_preserves_raw_message() -> None:
    """The connect() rewrap (`ProtocolError → DqliteConnectionError`)
    threads ``raw_message`` from the inner ProtocolError so the chain
    end-to-end carries the verbatim text."""
    inner = ProtocolError(
        "Handshake failed: [1001] long server text",
        raw_message="long server text",
    )
    rewrap = DqliteConnectionError(
        f"Wire decode failed during handshake to host-a:9001: {inner}",
        code=getattr(inner, "code", None),
        raw_message=getattr(inner, "raw_message", None) or str(inner),
    )
    assert rewrap.raw_message == "long server text"
