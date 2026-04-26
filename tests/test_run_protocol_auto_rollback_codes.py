"""When _run_protocol sees an OperationalError carrying a primary
SQLite code that implies server-side auto-rollback, _in_transaction
and _tx_owner must be cleared so the local state does not lie about
what the engine is doing.

The leader-class codes (NOT_LEADER, LEADERSHIP_LOST) trigger a full
_invalidate; codes in {SQLITE_ABORT, SQLITE_INTERRUPT, SQLITE_IOERR,
SQLITE_CORRUPT, SQLITE_FULL} only clear the tx flags — the connection
itself is still healthy.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


@pytest.mark.parametrize(
    "code,description",
    [
        (4, "SQLITE_ABORT"),
        (9, "SQLITE_INTERRUPT"),
        (10, "SQLITE_IOERR (primary)"),
        (10 | (8 << 8), "SQLITE_IOERR_FSTAT extended"),
        (11, "SQLITE_CORRUPT"),
        (13, "SQLITE_FULL"),
    ],
)
@pytest.mark.asyncio
async def test_auto_rollback_codes_clear_tx_flags(code: int, description: str) -> None:
    """Each auto-rollback primary code (or its extended variants)
    must clear _in_transaction / _tx_owner / savepoint stack /
    autobegin flag without invalidating the connection itself.
    Server-side SQLite engine has discarded the transaction including
    every savepoint frame; the local tracker must mirror that.
    """
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(code, f"simulated {description}")

    # Set tx + savepoint state so we can observe them being cleared.
    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp1", "sp2"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    # Connection must NOT be invalidated.
    assert conn._protocol is not None, f"{description}: connection unexpectedly invalidated"
    # All four state fields must be cleared atomically — mirror the
    # cleanup discipline already enforced by _invalidate / close.
    assert conn._in_transaction is False, f"{description}: _in_transaction still True"
    assert conn._tx_owner is None, f"{description}: _tx_owner not cleared"
    assert conn._savepoint_stack == [], f"{description}: _savepoint_stack not cleared"
    assert conn._savepoint_implicit_begin is False, (
        f"{description}: _savepoint_implicit_begin not cleared"
    )


@pytest.mark.parametrize(
    "code,description",
    [
        (5, "SQLITE_BUSY — application contention, tx still open"),
        (6, "SQLITE_LOCKED — table-level lock, tx still open"),
        (19, "SQLITE_CONSTRAINT — constraint violation, tx still open"),
    ],
)
@pytest.mark.asyncio
async def test_non_auto_rollback_codes_keep_tx_flags(code: int, description: str) -> None:
    """Codes outside the auto-rollback set leave the local tx flags
    intact — those errors do NOT cause the SQLite engine to clear the
    transaction."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(code, f"simulated {description}")

    owner = asyncio.current_task()
    conn._in_transaction = True
    conn._tx_owner = owner
    conn._savepoint_stack = ["sp1"]
    conn._savepoint_implicit_begin = False

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None
    # Flags preserved — user code must call ROLLBACK explicitly.
    # The savepoint stack and autobegin flag are also preserved
    # because the SQLite engine did NOT roll back the transaction.
    assert conn._in_transaction is True
    assert conn._tx_owner is owner
    assert conn._savepoint_stack == ["sp1"]
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_busy_with_checkpoint_message_clears_tx_state() -> None:
    """SQLITE_BUSY (5) with the upstream gateway's "checkpoint in
    progress" wording is a Raft-side BUSY where the tx-state-clear is
    safe — the in-flight write was not accepted, so the local tracker
    must mirror the server's view."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(5, "checkpoint in progress")

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None  # not invalidated
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
async def test_busy_with_engine_message_preserves_tx_state() -> None:
    """SQLITE_BUSY (5) with the standard SQLite-engine wording
    ("database is locked") is engine-side; the user can retry. The
    tracker stays in sync with the still-open tx."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(5, "database is locked")

    owner = asyncio.current_task()
    conn._in_transaction = True
    conn._tx_owner = owner
    conn._savepoint_stack = ["sp"]
    conn._savepoint_implicit_begin = False

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None
    assert conn._in_transaction is True
    assert conn._tx_owner is owner
    assert conn._savepoint_stack == ["sp"]
    assert conn._savepoint_implicit_begin is False


async def test_leader_class_codes_invalidate_and_clear_flags() -> None:
    """Leader-class codes still invalidate (existing behaviour) — the
    invalidate path itself clears the tx flags AND the savepoint stack
    AND the autobegin flag via the all-four-fields cleanup discipline."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        # SQLITE_IOERR_NOT_LEADER (10250) — leader class.
        raise OperationalError(10250, "not the leader")

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp1", "sp2"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    # Leader codes invalidate AND clear all four state fields.
    assert conn._protocol is None
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
