"""``transaction()`` does not invalidate on a COMMIT-arm failure with a deterministic-rollback
SQLite code (``_TX_AUTO_ROLLBACK_PRIMARY_CODES`` or ``SQLITE_CONSTRAINT`` 19): the server is
in a known no-tx state, so the connection stays healthy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError


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
    conn._protocol = MagicMock()  # appear connected so _check_connected does not raise
    conn._db_id = 1
    conn._check_in_use = lambda: None  # skip thread/loop binding
    return conn


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        7,  # SQLITE_NOMEM
        10,  # SQLITE_IOERR
        9,  # SQLITE_INTERRUPT
        11,  # SQLITE_CORRUPT
        13,  # SQLITE_FULL
        4,  # SQLITE_ABORT
        19,  # SQLITE_CONSTRAINT (deferred-FK on COMMIT)
    ],
)
async def test_transaction_commit_arm_does_not_invalidate_on_deterministic_rollback(
    code: int,
) -> None:
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise OperationalError("auto-rollback shape", code)

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(OperationalError):
        async with conn.transaction():
            pass

    invalidate_called.assert_not_called()


@pytest.mark.asyncio
async def test_transaction_commit_arm_invalidates_on_ambiguous_failure() -> None:
    """An ambiguous-shape commit-arm failure must still invalidate; discrimination is narrow."""
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            # Code 1 (SQLITE_ERROR) is not in the deterministic-rollback set; state is ambiguous.
            raise OperationalError("ambiguous error", 1)

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(OperationalError):
        async with conn.transaction():
            pass

    invalidate_called.assert_called_once()


@pytest.mark.asyncio
async def test_transaction_commit_arm_invalidates_on_cancelled_error() -> None:
    """CancelledError mid-COMMIT keeps server-side state ambiguous; invalidation is mandatory."""
    import asyncio

    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise asyncio.CancelledError()

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        async with conn.transaction():
            pass

    invalidate_called.assert_called_once()


@pytest.mark.asyncio
async def test_transaction_commit_arm_invalidates_on_extended_ioerr() -> None:
    """Extended IOERR 10250 maps to primary IOERR 10, in the rollback set; no invalidation."""
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise OperationalError("leader changed mid-commit", 10250)

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(OperationalError):
        async with conn.transaction():
            pass

    invalidate_called.assert_not_called()
