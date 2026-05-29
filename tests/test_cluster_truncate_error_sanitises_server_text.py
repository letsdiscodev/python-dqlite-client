"""``cluster._truncate_error`` runs the display sanitiser over ``raw_message`` (CWE-117).

Uses ``sanitize_server_text`` (not strict-escape) to preserve LF/Tab that SA's
``is_disconnect`` substring matcher expects in multi-line server diagnostics.
"""

from __future__ import annotations

from dqliteclient.cluster import _truncate_error


def test_strips_u2028_line_separator() -> None:
    raw = "error executing query INJECTED LOG LINE"
    out = _truncate_error(raw)
    assert " " not in out


def test_strips_bidi_override() -> None:
    raw = "benign‮malicious-rtl-content"
    out = _truncate_error(raw)
    assert "‮" not in out


def test_preserves_lf_for_multi_line_messages() -> None:
    raw = "line 1\nline 2"
    out = _truncate_error(raw)
    assert "\n" in out


def test_preserves_tab() -> None:
    raw = "col1\tcol2"
    out = _truncate_error(raw)
    assert "\t" in out


def test_short_message_passes_through_unchanged_after_sanitise() -> None:
    raw = "database is locked"
    out = _truncate_error(raw)
    assert out == raw


def test_substring_match_for_is_disconnect_preserved() -> None:
    """The sanitiser must not break SA's ``is_disconnect`` substring match on raw_message."""
    raw = "database is locked"
    out = _truncate_error(raw)
    assert "database is locked" in out
