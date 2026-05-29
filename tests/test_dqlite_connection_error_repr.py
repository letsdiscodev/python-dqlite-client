"""Pin: ``DqliteConnectionError.__repr__`` surfaces ``code`` when set so
forensic log lines (``logger.exception``, ``%r``) keep the wire-level
code. Mirrors the dbapi ``DatabaseError.__repr__`` discipline.
"""

from __future__ import annotations

from dqliteclient.exceptions import DqliteConnectionError


def test_dqlite_connection_error_repr_includes_code_when_set() -> None:
    e = DqliteConnectionError("Not leader", code=10250, raw_message="not leader")
    r = repr(e)
    assert "10250" in r, (
        f"DqliteConnectionError.__repr__ must include code=N when set; "
        f"got {r!r}. Mirrors the dbapi DatabaseError.__repr__ discipline."
    )
    assert "DqliteConnectionError" in r
    assert "'Not leader'" in r or '"Not leader"' in r


def test_dqlite_connection_error_repr_omits_code_when_none() -> None:
    """Negative twin: code=None renders without the noisy code= suffix."""
    e = DqliteConnectionError("Generic transport fault")
    r = repr(e)
    assert "DqliteConnectionError" in r
    assert "code=" not in r, (
        f"DqliteConnectionError.__repr__ must omit code= when code is None; got {r!r}"
    )


def test_dqlite_connection_error_repr_handles_empty_message() -> None:
    """A no-args DqliteConnectionError still reprs cleanly (message defaults to "")."""
    e = DqliteConnectionError(code=10250)
    r = repr(e)
    assert "DqliteConnectionError" in r
    assert "10250" in r
