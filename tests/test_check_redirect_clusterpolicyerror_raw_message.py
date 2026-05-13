"""``ClusterClient._check_redirect`` carries the verbatim peer-supplied
address on ``ClusterPolicyError.raw_message`` and renders the exception
text through ``_sanitize_display_text`` (control / bidi / invisible
stripped) — symmetric with the in-tree pattern at ``cluster.py:1162``
where ``_query_leader``'s malformed-redirect arm uses
``_sanitize_display_text(address)`` rather than Python's ``repr``.

The exception's ``args[0]`` used Python's generic ``!r`` formatter which
preserved Python escape sequences; the DEBUG sibling log used
``sanitize_for_log`` which used the wire-layer ``?`` substitution. The
asymmetry made cross-log correlation harder than necessary, and the
raw address was not available on ``raw_message`` for cross-process
forensic recovery (Celery worker -> result backend -> operator).
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_check_redirect_carries_raw_address_on_raw_message() -> None:
    """The verbatim peer-supplied address (with whatever LF / control
    bytes it carried) lands on ``exc.raw_message`` for cross-process
    forensic recovery. ``args[0]`` carries the sanitised display
    form."""
    cc = ClusterClient(
        MemoryNodeStore(["10.0.0.1:9001"]),
        timeout=5.0,
        redirect_policy=lambda _a: False,
    )
    hostile = "10.0.0.5:9001\n[ALERT] forged audit-trail line"
    with pytest.raises(ClusterPolicyError) as exc_info:
        cc._check_redirect(hostile)
    assert exc_info.value.raw_message == hostile


def test_check_redirect_exception_text_strips_control_chars() -> None:
    """Control characters (e.g. bidi overrides) in the address must not
    appear verbatim in the exception's ``args[0]`` — the wire-layer
    sanitiser substitutes ``?`` for them."""
    cc = ClusterClient(
        MemoryNodeStore(["10.0.0.1:9001"]),
        timeout=5.0,
        redirect_policy=lambda _a: False,
    )
    # U+202E is RIGHT-TO-LEFT OVERRIDE — a Trojan-Source-style bidi
    # character that ``_sanitize_display_text`` strips.
    hostile = "10.0.0.5:9001‮[trojan-source]"
    with pytest.raises(ClusterPolicyError) as exc_info:
        cc._check_redirect(hostile)
    assert "‮" not in str(exc_info.value)
    # Raw address is preserved verbatim on raw_message for forensics.
    assert exc_info.value.raw_message == hostile
