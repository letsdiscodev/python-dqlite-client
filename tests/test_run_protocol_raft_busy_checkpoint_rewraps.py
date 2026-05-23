"""Pin: ``_run_protocol`` rewraps a Raft-checkpoint SQLITE_BUSY
``OperationalError`` as ``DqliteConnectionError`` so SA's
``is_disconnect`` can classify via the connection-class arm.

The server emits ``code=SQLITE_BUSY (5)`` with the wording
"checkpoint in progress" when a Raft checkpoint reset the in-
flight transaction. Pre-fix the client cleared local tx flags and
re-raised the bare OperationalError; SA's ``is_disconnect`` gates
the substring scan on ``code is None`` so a coded BUSY was never
substring-scanned. SA's pool kept the slot, and SA's transaction
tracker diverged from server-side reality. Rewrapping as
DqliteConnectionError closes the gap.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError, OperationalError
from dqlitewire import SQLITE_BUSY


@pytest.mark.asyncio
async def test_run_protocol_rewraps_raft_busy_checkpoint_as_dqlite_connection_error() -> None:
    """An OperationalError(code=SQLITE_BUSY, message="...checkpoint
    in progress...") raised during _run_protocol must surface as
    DqliteConnectionError so SA's is_disconnect catches it."""
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
    # The original OperationalError is chained via __cause__ so the
    # full diagnostic is recoverable downstream.
    assert isinstance(ei.value.__cause__, OperationalError)
    # Local tx flags were cleared (existing contract).
    assert conn._in_transaction is False
    assert conn._tx_owner is None


@pytest.mark.asyncio
async def test_run_protocol_engine_busy_still_raises_operational_error() -> None:
    """Negative pin: a plain SQLITE_BUSY without the
    "checkpoint in progress" wording (i.e. SQLite-engine-side BUSY)
    is NOT rewrapped — the user can retry the statement on the
    same connection and continue the SAME transaction."""
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

    # Plain BUSY surfaces as OperationalError; the tx flags should
    # NOT be cleared (the user can retry within the same tx).
    with pytest.raises(OperationalError) as ei:
        await conn._run_protocol(_raising_fn)

    assert not isinstance(ei.value, DqliteConnectionError)
    assert conn._in_transaction is True  # tx still alive on engine-BUSY
