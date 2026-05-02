"""Pin: ``DqliteConnectionError`` accepts and preserves ``code`` and
``raw_message`` so a leader-change rewrap on the connect path
surfaces the wire-level signal that downstream classifiers expect.

When ``DqliteConnection.connect()`` catches a leader-change
``OperationalError(code=10250 / 10506, raw_message=...)`` from the
OPEN step, it rewraps the failure into ``DqliteConnectionError`` so
the dbapi-side classifier maps it to a transport-class error rather
than a SQL error. Without threading ``code`` and ``raw_message``
through the rewrap, both fields are dropped on the floor — SA's
``is_disconnect`` code-based branch can never fire on the connect
path, and any forensic reader who wants the verbatim server text
has to walk ``__cause__``.
"""

from __future__ import annotations

from dqliteclient.exceptions import DqliteConnectionError, DqliteError


def test_dqlite_connection_error_default_construction_works() -> None:
    """Backwards-compat: positional message still constructs an
    instance with code=None / raw_message=None."""
    e = DqliteConnectionError("Connection refused")
    assert str(e) == "Connection refused"
    assert e.code is None
    assert e.raw_message is None


def test_dqlite_connection_error_carries_code_and_raw_message() -> None:
    """A leader-change rewrap threads the wire-level diagnostic."""
    e = DqliteConnectionError(
        "Node leader-a:9001 is no longer leader: not leader",
        code=10250,
        raw_message="not leader",
    )
    assert e.code == 10250
    assert e.raw_message == "not leader"


def test_dqlite_connection_error_is_dqlite_error_subclass() -> None:
    """Defence pin against re-parenting."""
    assert issubclass(DqliteConnectionError, DqliteError)


def test_no_args_construction_works() -> None:
    """Some call sites raise with no message."""
    e = DqliteConnectionError()
    assert e.code is None
    assert e.raw_message is None
