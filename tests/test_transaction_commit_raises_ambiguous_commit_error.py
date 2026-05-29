"""``transaction()`` raises ``AmbiguousCommitError`` (an ``OperationalError`` subclass) when the
mid-flight COMMIT failure is not a deterministic-rollback shape, so retry middleware can branch
on "safe to retry" vs "may double-apply"."""

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
    """Transport failure mid-COMMIT surfaces as AmbiguousCommitError, original via __cause__."""
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
    """CancelledError mid-COMMIT propagates verbatim (not promoted), but still invalidates."""
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

    invalidate_called.assert_called_once()


@pytest.mark.asyncio
async def test_deterministic_rollback_code_does_not_raise_ambiguous_commit() -> None:
    """Deterministic-rollback code 19 (SQLITE_CONSTRAINT) stays a bare OperationalError."""
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

    assert not isinstance(excinfo.value, AmbiguousCommitError), (
        "deterministic rollback codes (SQLITE_CONSTRAINT etc.) must NOT "
        "promote to AmbiguousCommitError"
    )
    invalidate_called.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_except_operational_error_still_catches_ambiguous() -> None:
    """``except OperationalError`` still catches AmbiguousCommitError via subclassing."""
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
    assert isinstance(excinfo.value, AmbiguousCommitError)
