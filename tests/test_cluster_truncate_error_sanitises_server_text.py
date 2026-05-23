"""Pin: ``cluster._truncate_error`` composes the display sanitiser so
the value flowing into ``OperationalError.raw_message`` cannot carry
log-splitting characters (CWE-117 defence-in-depth).

Server-supplied text reaches this helper via every
``protocol._failure_text`` raise site (``raise OperationalError(...,
raw_message=response.message)``). ``raw_message`` is consumed by SA's
``is_disconnect`` substring matcher (safe -- byte-level scan) AND by
operator-side ``logger.error("%s", exc.raw_message)`` calls (NOT
safe without sanitisation).

The display variant (``sanitize_server_text``) is the right tool --
it preserves LF / Tab for multi-line server diagnostics and strips
control / bidi / invisible codepoints. Strict-escape would defeat
the substring-matcher's expectation that ``raw_message`` preserves
LF in multi-line server diagnostics.
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
    """A short, clean message must round-trip identically."""
    raw = "database is locked"
    out = _truncate_error(raw)
    assert out == raw


def test_substring_match_for_is_disconnect_preserved() -> None:
    """SA's ``is_disconnect`` substring matcher pre-scans
    ``raw_message`` for the wire-layer SQLite phrase 'database is
    locked'. Composing the display sanitiser must not break the
    substring-match expectation."""
    raw = "database is locked"
    out = _truncate_error(raw)
    assert "database is locked" in out
