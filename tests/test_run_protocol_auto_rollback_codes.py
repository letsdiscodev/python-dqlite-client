"""_run_protocol clears tx flags on auto-rollback SQLite codes; leader-class
codes additionally _invalidate, the rest leave the connection healthy."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


@pytest.mark.parametrize(
    "code,description",
    [
        (4, "SQLITE_ABORT"),
        (7, "SQLITE_NOMEM"),
        (9, "SQLITE_INTERRUPT"),
        (10, "SQLITE_IOERR (primary)"),
        (10 | (8 << 8), "SQLITE_IOERR_FSTAT extended"),
        (11, "SQLITE_CORRUPT"),
        (13, "SQLITE_FULL"),
    ],
)
@pytest.mark.asyncio
async def test_auto_rollback_codes_clear_tx_flags(code: int, description: str) -> None:
    """Each auto-rollback code clears tx + savepoint state without invalidating."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(f"simulated {description}", code)

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp1", "sp2"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None, f"{description}: connection unexpectedly invalidated"
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
    """Codes outside the auto-rollback set leave tx flags intact (engine keeps the tx)."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError(f"simulated {description}", code)

    owner = asyncio.current_task()
    conn._in_transaction = True
    conn._tx_owner = owner
    conn._savepoint_stack = ["sp1"]
    conn._savepoint_implicit_begin = False

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is not None
    assert conn._in_transaction is True
    assert conn._tx_owner is owner
    assert conn._savepoint_stack == ["sp1"]
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_busy_with_checkpoint_message_clears_tx_state() -> None:
    """Raft-checkpoint SQLITE_BUSY clears tx state and rewraps as
    DqliteConnectionError so SA's ``is_disconnect`` (gated on ``code is None``)
    catches it via the connection-class arm and the pool recycles the slot."""
    from dqliteclient.exceptions import DqliteConnectionError

    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError("checkpoint in progress", 5)

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(DqliteConnectionError) as ei:
        await conn._run_protocol(fake_send)

    assert isinstance(ei.value.__cause__, OperationalError)
    assert conn._protocol is not None  # not invalidated at the wire
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
async def test_busy_with_engine_message_preserves_tx_state() -> None:
    """Engine-side SQLITE_BUSY ("database is locked") leaves the tx open for retry."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError("database is locked", 5)

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
    """Leader-class codes invalidate; the invalidate path clears all tx state."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_send(protocol, db_id):
        raise OperationalError("not the leader", 10250)  # SQLITE_IOERR_NOT_LEADER

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp1", "sp2"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(OperationalError):
        await conn._run_protocol(fake_send)

    assert conn._protocol is None
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
async def test_busy_with_checkpoint_substring_only_in_raw_message_clears_tx_state() -> None:
    """The "checkpoint in progress" scan reads ``raw_message`` so a >1024-byte
    message whose wording is past ``message``'s truncation still classifies as
    Raft-BUSY."""
    from dqliteclient.exceptions import DqliteConnectionError

    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    # >1024-byte message: __init__ truncates ``message`` but keeps ``raw_message`` full.
    long_prefix = "context: " + ("X" * 1100)
    full = f"{long_prefix} checkpoint in progress"
    err = OperationalError(full, 5)
    assert "checkpoint in progress" not in err.message.lower()
    assert "checkpoint in progress" in err.raw_message.lower()

    async def fake_send(protocol: object, db_id: object) -> None:
        raise err

    conn._in_transaction = True
    conn._tx_owner = asyncio.current_task()
    conn._savepoint_stack = ["sp"]
    conn._savepoint_implicit_begin = True

    with pytest.raises(DqliteConnectionError) as ei:
        await conn._run_protocol(fake_send)

    assert isinstance(ei.value.__cause__, OperationalError)
    assert conn._protocol is not None  # not invalidated at the wire
    assert conn._in_transaction is False
    assert conn._tx_owner is None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
