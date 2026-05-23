"""Pin: ``client.OperationalError.__repr__`` produces a labelled
``code=N`` form, mirroring the sibling ``DqliteConnectionError`` and
the dbapi-layer ``OperationalError`` shape.

Without an explicit ``__repr__``, ``OperationalError`` inherited the
default ``BaseException.__repr__`` shape which renders the bare args
tuple — ``OperationalError('msg', 5)`` — producing inconsistent
formatting across the cause chain. Forensic log tooling that calls
``logger.error("%r", exc)`` on the unwrapped client-side error vs
the dbapi-wrapped sibling expects symmetric labelled output.
"""

from __future__ import annotations

from dqliteclient.exceptions import OperationalError


def test_operational_error_repr_includes_labelled_code() -> None:
    err = OperationalError("simulated failure", 5)
    rendered = repr(err)
    assert "OperationalError(" in rendered
    assert "'simulated failure'" in rendered
    assert "code=5" in rendered


def test_operational_error_repr_quotes_message_via_repr() -> None:
    """Messages with embedded quotes / escape chars must be repr'd
    (matching the sibling DqliteConnectionError discipline)."""
    err = OperationalError("line1\nline2", 12)
    rendered = repr(err)
    # The literal newline must NOT appear; the ``\\n`` escape must.
    assert "\n" not in rendered
    assert "\\n" in rendered
    assert "code=12" in rendered


def test_operational_error_repr_truncated_message_renders_truncated_form() -> None:
    """The display ``self.message`` (truncated at _MAX_DISPLAY_MESSAGE)
    is what appears in repr, NOT the raw_message — keeping repr bounded
    even for hostile-peer 64 KiB messages."""
    long_msg = "a" * 2000
    err = OperationalError(long_msg, 99)
    rendered = repr(err)
    # truncated form ends with "[truncated, N codepoints]"
    assert "[truncated" in rendered
    assert "code=99" in rendered
