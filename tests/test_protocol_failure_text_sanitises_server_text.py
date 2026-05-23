"""Pin: ``DqliteProtocol._failure_text`` routes the server-supplied
message through the display sanitiser (``sanitize_server_text``)
before composing the addr suffix.

Before this fix the body half of the exception display message
carried server text verbatim — U+2028 LINE SEPARATOR, U+2029, bidi
override chars, etc. — and a downstream ``logger.error("%s", exc)``
against journald could be log-split by a hostile peer's
``FailureResponse.message``. The sibling leader-flip rewrap arm in
``connection.py`` was already applying the same helper; this pins
the canonical query-path raise so both surfaces sanitise uniformly.

The display variant (``sanitize_server_text``) is the right tool —
it preserves LF / Tab for multi-line server diagnostics, strips
control / bidi / invisible codepoints. The strict
``sanitize_for_log`` variant would escape LF too, defeating the
multi-line server-message readability contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import FailureResponse


def _make_protocol() -> DqliteProtocol:
    return DqliteProtocol(MagicMock(), MagicMock(), timeout=5.0)


def test_failure_text_strips_u2028_line_separator() -> None:
    """U+2028 is allowed in TEXT decoding but splits journald records;
    the display sanitiser must strip it from the body half."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="error executing query INJECTED LOG LINE",
    )
    rendered = proto._failure_text(response)
    assert " " not in rendered


def test_failure_text_strips_bidi_override() -> None:
    """U+202E RIGHT-TO-LEFT OVERRIDE could reorder rendered output to
    hide attacker-controlled segments from operator scanning."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="benign ‮malicious-rtl-content",
    )
    rendered = proto._failure_text(response)
    assert "‮" not in rendered


def test_failure_text_preserves_lf_for_multi_line_messages() -> None:
    """Display variant must preserve LF so multi-line legitimate
    server diagnostics remain readable (not strict-escape semantics)."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="line 1\nline 2",
    )
    rendered = proto._failure_text(response)
    assert "\n" in rendered


def test_failure_text_preserves_tab_for_columnar_diagnostics() -> None:
    """Tab is preserved by the display variant — the strict log-bound
    variant escapes tabs; the display variant does not."""
    proto = _make_protocol()
    response = FailureResponse(
        code=1,
        message="col1\tcol2",
    )
    rendered = proto._failure_text(response)
    assert "\t" in rendered
