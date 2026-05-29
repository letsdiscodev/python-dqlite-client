"""``_ProbeMiss.message`` sanitizes the node address via sanitize_for_log
(LF/TAB-escaping), not the display-only sanitize_server_text: the message
reaches logger.error, and raw LF would split a log record (CWE-117).
"""

from __future__ import annotations


def _format_via_probe_miss(address: str, suffix: str) -> str:
    """Mirror the _probe_one timeout/no-leader/probe-failure string template."""
    from dqlitewire import sanitize_for_log

    return f"{sanitize_for_log(address)}: {suffix}"


def test_probe_miss_address_escapes_lf_for_log_embedding() -> None:
    """LF in node.address must be escaped to the literal ``\\n`` in the message."""
    forged = "127.0.0.1:9001\nFAKE LOG"
    msg = _format_via_probe_miss(forged, "timed out")
    assert "\n" not in msg, f"raw LF leaked into ProbeMiss.message: {msg!r}"
    assert "\\n" in msg, f"sanitize_for_log should escape LF; got {msg!r}"


def test_probe_miss_address_escapes_tab_for_log_embedding() -> None:
    """Same discipline for TAB."""
    forged = "127.0.0.1:9001\tFAKE LOG"
    msg = _format_via_probe_miss(forged, "no leader known")
    assert "\t" not in msg, f"raw TAB leaked into ProbeMiss.message: {msg!r}"


def test_probe_miss_address_strips_u2028() -> None:
    """U+2028 / U+2029 (journald record separators) are stripped."""
    forged = "127.0.0.1:9001 FAKE LOG"
    msg = _format_via_probe_miss(forged, "timed out")
    assert " " not in msg, f"U+2028 leaked into ProbeMiss.message: {msg!r}"
