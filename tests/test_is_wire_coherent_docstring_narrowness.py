"""Pin: ``DqliteProtocol.is_wire_coherent`` docstring scopes the
flag to codec-layer poison only and explicitly disclaims client-
layer ProtocolError raises.

The accessor reflects only ``MessageDecoder.is_poisoned``; client-
layer ProtocolError raises (db_id mismatch, extra-frame-after-FAILURE,
unexpected EmptyResponse mid-ROWS, interrupt drain caps,
_drain_continuations caps) leave it True at the moment of raise even
though the wire IS desynchronised at the next-request boundary. The
in-tree path is masked by _run_protocol's ProtocolError → _invalidate
chain; third-party harnesses on DqliteProtocol directly are not
masked.
"""

from __future__ import annotations

from dqliteclient.protocol import DqliteProtocol


def test_is_wire_coherent_docstring_calls_out_codec_layer_only() -> None:
    doc = DqliteProtocol.is_wire_coherent.__doc__ or ""
    assert "codec-layer" in doc.lower() or "codec layer" in doc.lower()
    assert "is_poisoned" in doc, "Docstring should name the underlying MessageDecoder.is_poisoned"


def test_is_wire_coherent_docstring_disclaims_client_layer_protocolerror() -> None:
    """The docstring must explicitly call out that the flag does NOT
    reflect client-layer ProtocolError raises."""
    doc = DqliteProtocol.is_wire_coherent.__doc__ or ""
    assert "Does NOT reflect" in doc or "does not reflect" in doc.lower(), (
        "Docstring must disclaim coverage of client-layer ProtocolError raises"
    )
    # The major sites the audit identified should be referenced so a
    # reader knows where to look for the contract drift.
    assert "_drain_continuations" in doc or "drain_continuations" in doc


def test_is_wire_coherent_docstring_directs_third_party_harnesses_to_invalidate() -> None:
    """Third-party direct DqliteProtocol harnesses must be told to
    invalidate on any ProtocolError rather than reading this flag."""
    doc = DqliteProtocol.is_wire_coherent.__doc__ or ""
    assert "third-party" in doc.lower() or "third party" in doc.lower()
    assert "invalidate" in doc.lower()
