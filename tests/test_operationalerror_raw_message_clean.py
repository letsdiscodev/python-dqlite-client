"""Pin: client ``OperationalError`` carries the verbatim
server text in ``raw_message`` (un-suffixed) — the cycle-21
contract that the dbapi layer plumbs through to its own
exceptions.

Pre-fix, ``protocol.py`` composed the addr-suffix into the
display message before calling ``OperationalError(code,
suffixed_message)``; the constructor copied the suffix-
contaminated text into ``raw_message``. The dbapi cursor
classifier then propagated the contamination verbatim,
breaking the "raw_message is the bytes the server actually
sent" invariant.
"""

from __future__ import annotations

from dqliteclient.exceptions import OperationalError


def test_operational_error_raw_message_keyword_preserves_server_text() -> None:
    """When the constructor is called with an explicit
    ``raw_message=`` kwarg, the verbatim server text is
    preserved while the display ``message`` carries the
    composed addr suffix."""
    err = OperationalError(
        5,
        "database is locked to localhost:9001",
        raw_message="database is locked",
    )
    assert err.message == "database is locked to localhost:9001"
    assert err.raw_message == "database is locked"
    assert "to localhost:9001" not in err.raw_message


def test_operational_error_default_raw_message_back_compat() -> None:
    """Old call sites that omit ``raw_message=`` still get the
    previous behaviour (``raw_message`` defaults to the
    display ``message``) so external callers do not break."""
    err = OperationalError(5, "database is locked")
    assert err.raw_message == "database is locked"


def test_operational_error_truncation_preserves_raw_message_full_length() -> None:
    """Display ``message`` is truncated at
    ``_MAX_DISPLAY_MESSAGE`` codepoints; ``raw_message``
    stays un-truncated for forensic / log-aggregator views."""
    long_text = "x" * 4096
    err = OperationalError(1, long_text, raw_message=long_text)
    assert len(err.raw_message) == 4096
    assert "[truncated," in err.message


def test_failure_text_truncates_message_before_appending_addr_suffix() -> None:
    """Pin: ``DqliteProtocol._failure_text`` truncates the server
    message BEFORE appending the addr suffix so the suffix
    survives the ``_MAX_DISPLAY_MESSAGE`` codepoint cap on the
    exception's display ``message`` field. Without pre-
    truncation, a 100k-char ORM-generated SQL error would
    push the addr suffix past the cutoff and operators
    tailing logs lose the peer-address attribution.
    """
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

    # Suffix must appear at the end of the rendered text.
    assert rendered.endswith(" to some-host:9001"), (
        f"Addr suffix must survive truncation; rendered ends with: {rendered[-100:]!r}"
    )
