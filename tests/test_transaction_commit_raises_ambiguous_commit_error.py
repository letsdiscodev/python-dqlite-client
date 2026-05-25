"""Pin: ``DqliteConnection.transaction()`` raises
``AmbiguousCommitError`` when the COMMIT mid-flight failure is not a
deterministic-rollback shape.

The transaction context manager already discriminates between
deterministic rollback (server-side auto-rolled-back; connection
healthy) and ambiguous commit (transport / cancellation /
non-rollback codes; commit state genuinely unknown). The ambiguous
branch invalidated the connection but re-raised the original
exception verbatim, leaving retry middleware unable to programmatic-
ally distinguish "safe to retry" from "may double-apply because
COMMIT may have already replicated." Mirror the dbapi-side
``AmbiguousCommitError`` so callers can branch on the class.

The class is a subclass of ``OperationalError`` so existing
``except OperationalError:`` arms continue to catch it.
"""

from __future__ import annotations

import asyncio
import pickle
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import (
    AmbiguousCommitError,
    DqliteConnectionError,
    OperationalError,
)


def _make_connection_in_transaction() -> DqliteConnection:
    from dqliteclient import connection as _conn_mod

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = "host:9001"
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop_ref = None
    conn._pending_drain = None
    conn._creator_pid = _conn_mod.get_current_pid()
    conn._pool_released = False
    conn._database = "main"
    conn._protocol = MagicMock()
    conn._db_id = 1
    conn._check_in_use = lambda: None
    return conn


def test_ambiguous_commit_error_is_subclass_of_operational_error() -> None:
    assert issubclass(AmbiguousCommitError, OperationalError)


def test_ambiguous_commit_pickles_losslessly() -> None:
    e = AmbiguousCommitError("ambig msg", 257, raw_message="raw payload")
    e2 = pickle.loads(pickle.dumps(e))
    assert isinstance(e2, AmbiguousCommitError)
    assert e2.code == 257
    assert e2.raw_message == "raw payload"
    assert str(e2) == "ambig msg"


@pytest.mark.asyncio
async def test_dqlite_connection_error_mid_commit_raises_ambiguous_commit() -> None:
    """A transport-class failure mid-COMMIT must surface as
    ``AmbiguousCommitError``. The original exception is preserved
    via ``__cause__``.
    """
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str, params=None) -> tuple[int, int]:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise DqliteConnectionError("transport gone mid-COMMIT")
        return (0, 0)

    conn.execute = _execute

    with pytest.raises(AmbiguousCommitError) as excinfo:
        async with conn.transaction():
            pass

    assert isinstance(excinfo.value.__cause__, DqliteConnectionError)
    assert "COMMIT mid-flight" in str(excinfo.value)
    invalidate_called.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_mid_commit_propagates_verbatim() -> None:
    """``CancelledError`` mid-COMMIT MUST propagate verbatim — it
    carries structured-concurrency semantics that asyncio relies on
    (TaskGroup teardown). The ambiguous-commit promotion explicitly
    skips ``CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit``.
    The connection is still invalidated (the server-side state IS
    ambiguous from the cancel), but the exception surface is the
    structured-concurrency signal.
    """
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str, params=None) -> tuple[int, int]:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise asyncio.CancelledError()
        return (0, 0)

    conn.execute = _execute

    with pytest.raises(asyncio.CancelledError):
        async with conn.transaction():
            pass

    # Connection is still invalidated — the server-side state is
    # genuinely ambiguous — but the exception surface preserves the
    # cancel signal.
    invalidate_called.assert_called_once()


@pytest.mark.asyncio
async def test_deterministic_rollback_code_does_not_raise_ambiguous_commit() -> None:
    """SQLite primary code 19 (SQLITE_CONSTRAINT) on COMMIT — a
    deterministic rollback shape — leaves the server in a known
    no-tx state. The exception must remain a bare
    ``OperationalError``, NOT promoted to ``AmbiguousCommitError``.
    """
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str, params=None) -> tuple[int, int]:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise OperationalError("constraint failed", 19)
        return (0, 0)

    conn.execute = _execute

    with pytest.raises(OperationalError) as excinfo:
        async with conn.transaction():
            pass

    # Must NOT be AmbiguousCommitError — deterministic rollback codes
    # are explicitly the "connection healthy, retry safe" path.
    assert not isinstance(excinfo.value, AmbiguousCommitError), (
        "deterministic rollback codes (SQLITE_CONSTRAINT etc.) must NOT "
        "promote to AmbiguousCommitError"
    )
    invalidate_called.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_except_operational_error_still_catches_ambiguous() -> None:
    """Backwards-compat pin: an existing call site that catches
    ``OperationalError`` continues to catch
    ``AmbiguousCommitError`` (via subclass inheritance).
    """
    conn = _make_connection_in_transaction()
    conn._invalidate = MagicMock()

    async def _execute(sql: str, params=None) -> tuple[int, int]:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise OperationalError("ambiguous error", 1)
        return (0, 0)

    conn.execute = _execute

    with pytest.raises(OperationalError) as excinfo:
        async with conn.transaction():
            pass
    # The promoted exception class is reachable via OperationalError —
    # the subclass relationship preserves backwards compat.
    assert isinstance(excinfo.value, AmbiguousCommitError)
