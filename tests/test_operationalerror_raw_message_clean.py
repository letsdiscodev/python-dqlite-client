"""``OperationalError.raw_message`` holds the verbatim server text without the addr suffix
that the display ``message`` carries."""

from __future__ import annotations

from dqliteclient.exceptions import OperationalError


def test_operational_error_raw_message_keyword_preserves_server_text() -> None:
    """Explicit ``raw_message=`` is preserved while ``message`` carries the addr suffix."""
    err = OperationalError(
        "database is locked to localhost:9001",
        5,
        raw_message="database is locked",
    )
    assert err.message == "database is locked to localhost:9001"
    assert err.raw_message == "database is locked"
    assert "to localhost:9001" not in err.raw_message


def test_operational_error_default_raw_message_back_compat() -> None:
    """Omitting ``raw_message=`` defaults it to the display ``message``."""
    err = OperationalError("database is locked", 5)
    assert err.raw_message == "database is locked"


def test_operational_error_truncation_preserves_raw_message_full_length() -> None:
    """Display ``message`` is truncated; ``raw_message`` stays un-truncated."""
    long_text = "x" * 4096
    err = OperationalError(long_text, 1, raw_message=long_text)
    assert len(err.raw_message) == 4096
    assert "[truncated," in err.message


def test_failure_text_truncates_message_before_appending_addr_suffix() -> None:
    """``_failure_text`` truncates BEFORE appending the addr suffix so the suffix survives
    the display-message cap."""
    from unittest.mock import AsyncMock, MagicMock

    from dqliteclient.protocol import DqliteProtocol
    from dqlitewire.messages import FailureResponse

    proto = DqliteProtocol(
        AsyncMock(spec=["read"]),
        MagicMock(spec=["close", "wait_closed"]),
        address="some-host:9001",
    )
    huge = "x" * 100_000
    response = FailureResponse(code=1, message=huge)

    rendered = proto._failure_text(response)

    assert rendered.endswith(" to some-host:9001"), (
        f"Addr suffix must survive truncation; rendered ends with: {rendered[-100:]!r}"
    )
