"""Pin: ``DqliteConnectionError.__repr__`` surfaces the ``code``
attribute when set so forensic log lines (``logger.exception``,
``logger.error("%r", exc)``, ``pytest``'s exception-with-args display)
do not silently drop the wire-level code.

Mirrors the dbapi sibling at
``python-dqlite-dbapi/src/dqlitedbapi/exceptions.py::DatabaseError.__repr__``
which renders ``DatabaseError('msg', code=N)`` when ``code is not
None``. The client-layer ``DqliteConnectionError`` carries the
exact same ``code`` attribute (set on the connect-path rewrap of an
upstream ``OperationalError(LEADER_ERROR_CODES)``), so the repr
discipline must be symmetric across the two layers.

Scope is JUST ``DqliteConnectionError``. Other client-layer
``DqliteError`` siblings (``OperationalError``, ``ProtocolError``,
``DataError``, ``InterfaceError``) are unchanged: only
``OperationalError`` carries a ``code`` attribute and its repr is
already handled by ``Exception.__repr__`` rendering the ``args``
tuple ``(message, code)`` directly.
"""

from __future__ import annotations

from dqliteclient.exceptions import DqliteConnectionError


def test_dqlite_connection_error_repr_includes_code_when_set() -> None:
    """When ``code`` is non-None, ``repr(exc)`` must surface it so
    forensic log lines (``logger.exception``,
    ``logger.error("%r", exc)``) capture the wire-level code rather
    than dropping it.
    """
    e = DqliteConnectionError("Not leader", code=10250, raw_message="not leader")
    r = repr(e)
    assert "10250" in r, (
        f"DqliteConnectionError.__repr__ must include code=N when set; "
        f"got {r!r}. Mirrors the dbapi DatabaseError.__repr__ discipline."
    )
    assert "DqliteConnectionError" in r
    assert "'Not leader'" in r or '"Not leader"' in r


def test_dqlite_connection_error_repr_omits_code_when_none() -> None:
    """Negative twin: when ``code`` is None (default), repr renders
    without the ``code=`` suffix so a transport-only fault doesn't
    surface a noisy ``code=None`` line.
    """
    e = DqliteConnectionError("Generic transport fault")
    r = repr(e)
    assert "DqliteConnectionError" in r
    assert "code=" not in r, (
        f"DqliteConnectionError.__repr__ must omit code= when code is None; got {r!r}"
    )


def test_dqlite_connection_error_repr_handles_empty_message() -> None:
    """Defensive: a no-args ``DqliteConnectionError`` must still
    repr cleanly (the constructor defaults message to ``""``).
    """
    e = DqliteConnectionError(code=10250)
    r = repr(e)
    assert "DqliteConnectionError" in r
    assert "10250" in r
