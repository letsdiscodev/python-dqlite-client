"""Pin: ``cluster._truncate_error`` reports the overflow count
(``len - max``) in its ``[truncated, N chars]`` suffix, matching the
wire-layer ``_cap_raw_message`` SSOT and the sibling
``_truncate_for_message`` / ``_truncate_for_log`` helpers in dbapi
and SA.

A prior version of ``_truncate_error`` reported the total length
(``len(message)``) — operator-confusing because a reader seeing
``[truncated, 4096 chars]`` would assume the original was 4096
characters when the actual size is ``max + 4096``.
"""

from __future__ import annotations

from dqliteclient.cluster import _MAX_ERROR_MESSAGE_SNIPPET, _truncate_error


def test_truncate_error_suffix_reports_overflow_not_total() -> None:
    overflow = 256
    message = "x" * (_MAX_ERROR_MESSAGE_SNIPPET + overflow)

    result = _truncate_error(message)

    assert result.endswith(f"... [truncated, {overflow} chars]")
    # The total-length number must NOT appear (would be the old
    # convention).
    total = _MAX_ERROR_MESSAGE_SNIPPET + overflow
    assert f"[truncated, {total} chars]" not in result


def test_truncate_error_under_cap_unchanged() -> None:
    message = "x" * (_MAX_ERROR_MESSAGE_SNIPPET - 1)
    assert _truncate_error(message) == message


def test_truncate_error_at_cap_unchanged() -> None:
    message = "x" * _MAX_ERROR_MESSAGE_SNIPPET
    assert _truncate_error(message) == message
