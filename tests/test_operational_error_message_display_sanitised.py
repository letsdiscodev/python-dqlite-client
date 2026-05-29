"""``OperationalError.__init__`` scrubs control bytes from the display ``message`` at the
class boundary to prevent log injection (CWE-117); ``raw_message`` stays verbatim."""

from __future__ import annotations

import pytest

from dqliteclient.exceptions import OperationalError


def test_operational_error_message_strips_cr() -> None:
    """Scrubs CR (and most control bytes) but intentionally preserves LF and Tab."""
    e = OperationalError("hello\rWARNING: faked log line", 1)
    assert "\r" not in str(e)


def test_operational_error_message_strips_nul() -> None:
    e = OperationalError("hello\x00world", 1)
    assert "\x00" not in str(e)


def test_operational_error_message_strips_ansi_escape() -> None:
    e = OperationalError("\x1b[31mred\x1b[0m", 1)
    assert "\x1b" not in str(e)


def test_operational_error_raw_message_preserves_unsanitised_text() -> None:
    """``raw_message`` stays verbatim; only the display field is sanitised."""
    raw = "hello\r\nWARNING\x00\x1b[31m"
    e = OperationalError("display", 1, raw_message=raw)
    assert e.raw_message == raw


def test_operational_error_raw_message_defaults_to_unsanitised_message() -> None:
    """When omitted, ``raw_message`` is populated from ``message`` BEFORE sanitisation."""
    raw = "hello\r\nWARNING\x00"
    e = OperationalError(raw, 1)
    assert e.raw_message == raw


def test_operational_error_message_idempotent_pre_sanitised() -> None:
    """Re-sanitising already-clean input is idempotent."""
    clean = "ordinary diagnostic"
    e = OperationalError(clean, 1)
    assert str(e) == clean


@pytest.mark.parametrize("ctrl", ["\x07", "\x0c", "\x1f"])
def test_operational_error_message_strips_other_control_bytes(ctrl: str) -> None:
    e = OperationalError(f"prefix{ctrl}suffix", 1)
    assert ctrl not in str(e)
