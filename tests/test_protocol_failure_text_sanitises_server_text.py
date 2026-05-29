"""Pin: ``DqliteProtocol._failure_text`` sanitises server text via
the display variant (strips control/bidi, keeps LF/Tab) so a hostile
``FailureResponse.message`` cannot log-split journald downstream.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


def _make_protocol() -> DqliteProtocol:
    return DqliteProtocol(MagicMock(), MagicMock(), timeout=5.0)


def test_failure_text_strips_u2028_line_separator() -> None:
    """U+2028 is legal in TEXT decoding but splits journald records."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="error executing query INJECTED LOG LINE",
    )
    rendered = proto._failure_text(response)
    assert " " not in rendered


def test_failure_text_strips_bidi_override() -> None:
    """U+202E RIGHT-TO-LEFT OVERRIDE could hide attacker segments."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="benign ‮malicious-rtl-content",
    )
    rendered = proto._failure_text(response)
    assert "‮" not in rendered


def test_failure_text_preserves_lf_for_multi_line_messages() -> None:
    """Display variant preserves LF for multi-line server diagnostics."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="line 1\nline 2",
    )
    rendered = proto._failure_text(response)
    assert "\n" in rendered


def test_failure_text_preserves_tab_for_columnar_diagnostics() -> None:
    """Tab is preserved by the display variant (the strict one escapes it)."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="col1\tcol2",
    )
    rendered = proto._failure_text(response)
    assert "\t" in rendered
