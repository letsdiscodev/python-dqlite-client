"""``_check_redirect`` carries the verbatim peer address on
``ClusterPolicyError.raw_message`` (for cross-process forensics) and renders
``args[0]`` through ``_sanitize_display_text`` (control/bidi stripped)."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_check_redirect_carries_raw_address_on_raw_message() -> None:
    """The verbatim peer address (LF/control bytes intact) lands on
    ``exc.raw_message``; ``args[0]`` carries the sanitised form."""
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
    """Control characters in the address are sanitised out of ``args[0]``."""
    cc = ClusterClient(
        MemoryNodeStore(["10.0.0.1:9001"]),
        timeout=5.0,
        redirect_policy=lambda _a: False,
    )
    # Embedded char is U+202E RIGHT-TO-LEFT OVERRIDE (Trojan-Source bidi).
    hostile = "10.0.0.5:9001‮[trojan-source]"
    with pytest.raises(ClusterPolicyError) as exc_info:
        cc._check_redirect(hostile)
    assert "‮" not in str(exc_info.value)
    assert exc_info.value.raw_message == hostile
