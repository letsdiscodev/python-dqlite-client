"""Pin: handshake-`FailureResponse` `OperationalError` raise preserves
``raw_message=response.message`` (the verbatim 64-KiB-capped server
text) AND the structured ``code`` — and the surrounding ``connect()``
rewrap to ``DqliteConnectionError`` carries those fields through.

The 16 sibling RPC raise sites in protocol.py thread
``OperationalError(text, code, raw_message=response.message)`` so
cross-process forwarding (multiprocessing.Queue) and SA-layer
classification have the verbatim diagnostic text plus the structured
code for routing. The handshake site was the lone outlier emitting a
ProtocolError with the code only interpolated into the message text;
this pin covers the end-to-end chain: wire layer → OperationalError →
DqliteConnectionError.
"""

from __future__ import annotations

from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


def _make_proto() -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._address = "host-a:9001"
    return proto


def test_handshake_failure_response_operational_error_preserves_raw_message_and_code() -> None:
    """A FailureResponse during handshake produces an OperationalError
    whose ``raw_message`` matches the verbatim server text and whose
    ``code`` attribute equals the wire code, mirroring the 16 sibling
    RPC raise sites."""
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
    """The connect() rewrap (`OperationalError → DqliteConnectionError`)
    threads ``raw_message`` from the inner OperationalError so the
    chain end-to-end carries the verbatim text."""
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
