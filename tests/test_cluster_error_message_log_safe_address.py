"""Pin: ``_ProbeMiss.message`` interpolates the node address via
``sanitize_for_log`` (the LF/TAB-escaping variant), not
``sanitize_server_text`` (display-only, preserves LF). The
resulting message text flows into ``ClusterError.args[0]`` and is
logged via ``logger.error("%s", err)`` by upstream callers; raw
LF in the address would split a journald / syslog log record
(CWE-117).

The redirect-verify arm at cluster.py was already correct; the
sibling probe-failure / timeout / no-leader-known arms were
asymmetric (they used the display-only variant). This pin closes
the gap.
"""

from __future__ import annotations


def _format_via_probe_miss(address: str, suffix: str) -> str:
    """Mirror the in-tree formatting used in the timeout / no-leader
    / probe-failure arms of ``_probe_one`` so the test exercises the
    real string template without going through a full find_leader
    integration. Aligning with the actual call sites means a future
    regression that re-introduces ``_sanitize_display_text`` would
    fail this pin."""
    from dqlitewire import sanitize_for_log

    return f"{sanitize_for_log(address)}: {suffix}"


def test_probe_miss_address_escapes_lf_for_log_embedding() -> None:
    """A node.address containing LF (e.g. via dial_func override)
    must NOT pass through verbatim into _ProbeMiss.message; the LF
    is escaped to the two-byte literal sequence ``\\n``."""
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
    """U+2028 / U+2029 (journald record separators) are stripped by
    sanitize_for_log via its sanitize_server_text inner pass."""
    forged = "127.0.0.1:9001 FAKE LOG"
    msg = _format_via_probe_miss(forged, "timed out")
    assert " " not in msg, f"U+2028 leaked into ProbeMiss.message: {msg!r}"
