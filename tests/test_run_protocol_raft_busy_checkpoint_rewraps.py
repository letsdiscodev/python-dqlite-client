"""``_run_protocol`` rewraps a Raft-checkpoint SQLITE_BUSY ("checkpoint in progress")
as ``DqliteConnectionError`` so SA's ``is_disconnect`` (which gates its substring scan
on ``code is None``) classifies it via the connection-class arm and recycles the slot."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqlitewire import SQLITE_BUSY


@pytest.mark.asyncio
async def test_run_protocol_rewraps_raft_busy_checkpoint_as_dqlite_connection_error() -> None:
    """A checkpoint-in-progress SQLITE_BUSY must surface as DqliteConnectionError."""
    import os

    conn = DqliteConnection.__new__(DqliteConnection)
    import asyncio as _asyncio

    conn._in_transaction = True
    conn._tx_owner = _asyncio.current_task()
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._in_use = False
    conn._invalidation_cause = None
    conn._creator_pid = os.getpid()
    conn._pool_released = False
    conn._bound_loop_ref = None
    conn._protocol = MagicMock()
    conn._protocol._writer = MagicMock()
    conn._db_id = 7
    conn._close_timeout = 1.0
    conn._address = "127.0.0.1:9001"

    async def _raising_fn(_protocol: object, _db_id: object) -> object:
        raise OperationalError(
            "checkpoint in progress; transaction aborted",
            code=SQLITE_BUSY,
            raw_message="checkpoint in progress; transaction aborted",
        )

    with pytest.raises(DqliteConnectionError) as ei:
        await conn._run_protocol(_raising_fn)

    assert "checkpoint" in str(ei.value).lower()
    assert isinstance(ei.value.__cause__, OperationalError)
    assert conn._in_transaction is False
    assert conn._tx_owner is None


@pytest.mark.asyncio
async def test_run_protocol_engine_busy_still_raises_operational_error() -> None:
    """A plain engine-side SQLITE_BUSY is NOT rewrapped; the tx stays open for retry."""
    import os

    conn = DqliteConnection.__new__(DqliteConnection)
    import asyncio as _asyncio

    conn._in_transaction = True
    conn._tx_owner = _asyncio.current_task()
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._in_use = False
    conn._invalidation_cause = None
    conn._creator_pid = os.getpid()
    conn._pool_released = False
    conn._bound_loop_ref = None
    conn._protocol = MagicMock()
    conn._protocol._writer = MagicMock()
    conn._db_id = 7
    conn._close_timeout = 1.0
    conn._address = "127.0.0.1:9001"

    async def _raising_fn(_protocol: object, _db_id: object) -> object:
        raise OperationalError(
            "database is locked",
            code=SQLITE_BUSY,
            raw_message="database is locked",
        )

    with pytest.raises(OperationalError) as ei:
        await conn._run_protocol(_raising_fn)

    assert not isinstance(ei.value, DqliteConnectionError)
    assert conn._in_transaction is True  # tx still alive on engine-BUSY
