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
    must clear _in_transaction / _tx_owner without invalidating
    the connection itself."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(code, f"simulated {description}")

    # Set tx flags so we can observe them being cleared.
    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    # Connection must NOT be invalidated.
    assert conn._protocol is not None, f"{description}: connection unexpectedly invalidated"
    # tx flags must be cleared.
    assert conn._in_transaction is False, f"{description}: _in_transaction still True"
    assert conn._tx_owner is None, f"{description}: _tx_owner not cleared"


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

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None
    # Flags preserved — user code must call ROLLBACK explicitly.
    assert conn._in_transaction is True
    assert conn._tx_owner is owner


@pytest.mark.asyncio
async def test_leader_class_codes_invalidate_and_clear_flags() -> None:
    """Leader-class codes still invalidate (existing behaviour) — the
    invalidate path itself clears the tx flags via ISSUE-693."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        # SQLITE_IOERR_NOT_LEADER (10250) — leader class.
        raise OperationalError(10250, "not the leader")

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    # Leader codes invalidate.
    assert conn._protocol is None
    assert conn._in_transaction is False
    assert conn._tx_owner is None
