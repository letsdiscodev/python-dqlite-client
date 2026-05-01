"""Pin: ``DqliteProtocol`` deliberately does NOT invoke
``skip_message`` to recover from oversize-message rejections — every
wire-level ProtocolError (including the recoverable, non-poisoning
oversize DecodeError) is wrapped to a client-level ProtocolError and
routed through ``_invalidate`` to drop the connection.

The wire layer documents oversize as a recoverable-in-place
affordance (``read_message`` raises without poisoning; ``skip_message``
drains). The client layer opts out for security (a 64 MiB cap is
high enough that hitting it almost always indicates an attacker, a
misbehaving server, or a wire bug — none safe to resume from on the
same socket). Pin so a future contributor reading the wire-layer
docstring doesn't quietly add ``skip_message`` calls into the client
without re-evaluating the security posture.
"""

from __future__ import annotations

import inspect

from dqliteclient import protocol as proto_mod


def test_protocol_module_does_not_invoke_skip_message() -> None:
    """Source-level pin: no client-layer code path calls
    ``skip_message`` or reads ``is_skipping``. A regression that
    re-introduces the recovery would show up here."""
    src = inspect.getsource(proto_mod)
    assert "skip_message" not in src or "skip_message`` API" in src, (
        "DqliteProtocol must not invoke skip_message — wire-level "
        "DecodeError must drop the connection per the deliberate "
        "opt-out documented in the _read_response wrap site."
    )
    assert "is_skipping" not in src, (
        "DqliteProtocol must not read is_skipping — the skip-recovery path is unused by design."
    )


def test_protocol_documents_skip_message_opt_out_rationale() -> None:
    """The ``_read_response`` wrap site (or its sibling) must contain
    a comment block documenting the deliberate opt-out, so a future
    reader of the wire layer's recoverable-oversize docstring is
    redirected to the client-layer policy instead of quietly adding
    skip_message calls."""
    src = inspect.getsource(proto_mod)
    assert "deliberately opts out" in src or "deliberate opt-out" in src.lower(), (
        "The protocol module should document the skip_message opt-out "
        "near the wire-DecodeError wrap site"
    )
    assert "skip_message" in src, (
        "The opt-out comment should name skip_message so a future reader can find it via grep"
    )
