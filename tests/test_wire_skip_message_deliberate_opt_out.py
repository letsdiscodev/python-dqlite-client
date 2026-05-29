"""Pin: ``DqliteProtocol`` deliberately does NOT invoke ``skip_message`` to recover
from oversize rejections; it drops the connection instead.

Hitting the 64 MiB cap almost always means an attacker, a misbehaving server, or a
wire bug, none safe to resume from on the same socket. The wire layer offers
oversize as recoverable-in-place; the client opts out for security.
"""

from __future__ import annotations

import inspect

from dqliteclient import protocol as proto_mod


def test_protocol_module_does_not_invoke_skip_message() -> None:
    """Source-level pin: no client-layer code path calls skip_message or reads
    is_skipping."""
    src = inspect.getsource(proto_mod)
    assert "skip_message(" not in src, (
        "DqliteProtocol must not invoke skip_message — wire-level "
        "DecodeError must drop the connection per the deliberate opt-out."
    )
    assert "is_skipping" not in src, (
        "DqliteProtocol must not read is_skipping — the skip-recovery path is unused by design."
    )
