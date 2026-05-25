"""Pin: ``OperationalError.__init__`` display-sanitises ``message``
at the class boundary, regardless of whether the caller pre-sanitised.

Pre-fix the constructor truncated to ``_MAX_DISPLAY_MESSAGE``
codepoints but did NOT scrub CR / LF / NUL / ANSI escape / other
control bytes — the contract "callers sanitise before passing" lived
implicitly at every first-party call site. A new call site (or a
third-party caller raising ``OperationalError(server_text, code)``
directly) would leak raw control bytes into ``str(e)`` and onward to
log handlers / SIEM / SQLAlchemy's ``is_disconnect`` substring scan
(log-injection class CWE-117).

``raw_message`` continues to carry the verbatim bytes for forensic
recovery; only the display field is sanitised.
"""

from __future__ import annotations

import pytest

from dqliteclient.exceptions import OperationalError


def test_operational_error_message_strips_cr() -> None:
    """Sanitiser scrubs CR (and most control bytes) but intentionally
    preserves LF and Tab so multi-line server diagnostics render
    correctly. ``sanitize_for_log`` is the log-callsite helper that
    further escapes LF; the exception field stays human-readable.
    """
    e = OperationalError("hello\rWARNING: faked log line", 1)
    assert "\r" not in str(e)


def test_operational_error_message_strips_nul() -> None:
    e = OperationalError("hello\x00world", 1)
    assert "\x00" not in str(e)


def test_operational_error_message_strips_ansi_escape() -> None:
    e = OperationalError("\x1b[31mred\x1b[0m", 1)
    assert "\x1b" not in str(e)


def test_operational_error_raw_message_preserves_unsanitised_text() -> None:
    """Forensic-recovery field stays verbatim — the sanitisation only
    applies to the display field. Callers chasing the original peer
    bytes (e.g. dbapi classifier, cross-process pickled exception
    graphs) still see what the wire delivered.
    """
    raw = "hello\r\nWARNING\x00\x1b[31m"
    e = OperationalError("display", 1, raw_message=raw)
    assert e.raw_message == raw


def test_operational_error_raw_message_defaults_to_unsanitised_message() -> None:
    """When ``raw_message`` is omitted, ``DqliteError`` populates it
    from the constructor's ``message`` argument BEFORE sanitisation,
    so the verbatim peer text survives even if the operator passes
    only ``message=``.
    """
    raw = "hello\r\nWARNING\x00"
    e = OperationalError(raw, 1)
    assert e.raw_message == raw


def test_operational_error_message_idempotent_pre_sanitised() -> None:
    """Existing first-party call sites already sanitise before
    constructing the exception. Applying ``sanitize_server_text``
    again is idempotent — already-safe input is unchanged.
    """
    clean = "ordinary diagnostic"
    e = OperationalError(clean, 1)
    assert str(e) == clean


@pytest.mark.parametrize("ctrl", ["\x07", "\x0c", "\x1f"])
def test_operational_error_message_strips_other_control_bytes(ctrl: str) -> None:
    e = OperationalError(f"prefix{ctrl}suffix", 1)
    assert ctrl not in str(e)
