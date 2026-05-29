"""``OperationalError.__repr__`` produces a labelled ``code=N`` form, matching the sibling
``DqliteConnectionError`` and dbapi ``OperationalError``."""

from __future__ import annotations

from dqliteclient.exceptions import OperationalError


def test_operational_error_repr_includes_labelled_code() -> None:
    err = OperationalError("simulated failure", 5)
    rendered = repr(err)
    assert "OperationalError(" in rendered
    assert "'simulated failure'" in rendered
    assert "code=5" in rendered


def test_operational_error_repr_quotes_message_via_repr() -> None:
    """Messages with embedded escape chars must be repr'd, not rendered literally."""
    err = OperationalError("line1\nline2", 12)
    rendered = repr(err)
    assert "\n" not in rendered
    assert "\\n" in rendered
    assert "code=12" in rendered


def test_operational_error_repr_truncated_message_renders_truncated_form() -> None:
    """repr uses the truncated display ``message``, not ``raw_message``, so it stays bounded."""
    long_msg = "a" * 2000
    err = OperationalError(long_msg, 99)
    rendered = repr(err)
    assert "[truncated" in rendered
    assert "code=99" in rendered
