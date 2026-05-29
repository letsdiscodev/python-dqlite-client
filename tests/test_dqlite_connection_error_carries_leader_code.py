"""Pin: ``DqliteConnectionError`` preserves ``code`` and ``raw_message``
so the connect-path leader-change rewrap keeps the wire-level signal
SA's ``is_disconnect`` code branch and forensic readers depend on.
"""

from __future__ import annotations

from dqliteclient.exceptions import DqliteConnectionError, DqliteError


def test_dqlite_connection_error_default_construction_works() -> None:
    """Backwards-compat: positional message gives code=None / raw_message=None."""
    e = DqliteConnectionError("Connection refused")
    assert str(e) == "Connection refused"
    assert e.code is None
    assert e.raw_message is None


def test_dqlite_connection_error_carries_code_and_raw_message() -> None:
    e = DqliteConnectionError(
        "Node leader-a:9001 is no longer leader: not leader",
        code=10250,
        raw_message="not leader",
    )
    assert e.code == 10250
    assert e.raw_message == "not leader"


def test_dqlite_connection_error_is_dqlite_error_subclass() -> None:
    assert issubclass(DqliteConnectionError, DqliteError)


def test_no_args_construction_works() -> None:
    e = DqliteConnectionError()
    assert e.code is None
    assert e.raw_message is None
