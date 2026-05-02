"""Pin: ``DqliteConnection.transaction()`` does NOT invalidate the
connection on a COMMIT-arm failure whose SQLite primary code is a
deterministic-rollback shape (``_TX_AUTO_ROLLBACK_PRIMARY_CODES`` or
``SQLITE_CONSTRAINT`` = 19).

Both code classes leave the server in a known no-tx state — the
local tx flags were already cleared by ``execute`` /
``_run_protocol`` before the exception propagated. Invalidating
discards a healthy connection, forcing a fresh-connect round-trip
under retry storms. Mirrors the rollback-arm discrimination already
in place at the sibling no-tx-rollback path (done/tx-013).
"""

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
    conn._bound_loop = None
    conn._pending_drain = None
    conn._creator_pid = _conn_mod._current_pid
    conn._pool_released = False
    conn._database = "main"
    # Mark as connected so _check_connected does not raise.
    conn._protocol = MagicMock()
    conn._db_id = 1
    # Suppress thread/loop binding by short-circuiting _check_in_use.
    conn._check_in_use = lambda: None
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
    """Drive the transaction ctxmgr to the commit-arm failure path
    with a deterministic-rollback OperationalError code. The
    invalidate path must NOT fire."""
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    # Mock execute so BEGIN succeeds, COMMIT raises with the code.
    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise OperationalError(code, "auto-rollback shape")

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(OperationalError):
        async with conn.transaction():
            pass

    invalidate_called.assert_not_called()


@pytest.mark.asyncio
async def test_transaction_commit_arm_invalidates_on_ambiguous_failure() -> None:
    """Defence pin: a commit-arm failure with an ambiguous shape
    (e.g. CancelledError, transport error, non-rollback OperationalError)
    MUST still invalidate. The discrimination is narrow."""
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            # Code 1 (SQLITE_ERROR) is NOT in the deterministic-
            # rollback set; server-side state is ambiguous.
            raise OperationalError(1, "ambiguous error")

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(OperationalError):
        async with conn.transaction():
            pass

    invalidate_called.assert_called_once()


@pytest.mark.asyncio
async def test_transaction_commit_arm_invalidates_on_cancelled_error() -> None:
    """CancelledError mid-COMMIT keeps server-side state ambiguous —
    invalidation is still mandatory."""
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
    """An extended IOERR code (10250 = SQLITE_IOERR_NOT_LEADER) maps
    to primary IOERR (10) which IS in the rollback set; pin that the
    extended-code arithmetic doesn't trip the discrimination."""
    conn = _make_connection_in_transaction()
    invalidate_called = MagicMock()
    conn._invalidate = invalidate_called

    async def _execute(sql: str) -> None:
        if "COMMIT" in sql.upper() or sql.strip().upper() == "END":
            raise OperationalError(10250, "leader changed mid-commit")

    conn.execute = _execute  # type: ignore[assignment]

    with pytest.raises(OperationalError):
        async with conn.transaction():
            pass

    # IOERR's primary is in the deterministic-rollback set; the
    # connection should NOT be invalidated.
    invalidate_called.assert_not_called()
