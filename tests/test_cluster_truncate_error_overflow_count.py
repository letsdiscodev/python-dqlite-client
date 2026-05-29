"""``cluster._truncate_error`` reports the overflow count (len - max), not the total length."""

from __future__ import annotations

from dqliteclient.cluster import _MAX_ERROR_MESSAGE_SNIPPET, _truncate_error


def test_truncate_error_suffix_reports_overflow_not_total() -> None:
    overflow = 256
    message = "x" * (_MAX_ERROR_MESSAGE_SNIPPET + overflow)

    result = _truncate_error(message)

    assert result.endswith(f"... [truncated, {overflow} chars]")
    total = _MAX_ERROR_MESSAGE_SNIPPET + overflow
    assert f"[truncated, {total} chars]" not in result


def test_truncate_error_under_cap_unchanged() -> None:
    message = "x" * (_MAX_ERROR_MESSAGE_SNIPPET - 1)
    assert _truncate_error(message) == message


def test_truncate_error_at_cap_unchanged() -> None:
    message = "x" * _MAX_ERROR_MESSAGE_SNIPPET
    assert _truncate_error(message) == message
