"""Pin: ``DqliteConnection.close()``'s fork-pid shortcut clears
``self._in_use`` along with every other lifecycle field.

A forked-after-init worker that inherits ``_in_use=True`` would
otherwise be permanently locked out behind ``_check_in_use``'s
"another operation is in progress" diagnostic. Every other lifecycle
field cleared on this branch follows the same "drop parent-loop
state" rationale; ``_in_use`` was the lone omission until this pin.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


def _make_connection() -> DqliteConnection:
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._closed = False
    conn._closed_flag = [False]
    conn._protocol = MagicMock()
    conn._db_id = 1
    conn._pending_drain = None
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._bound_loop_ref = None
    conn._finalizer = None
    conn._pool_released = False
    return conn


@pytest.mark.asyncio
async def test_close_fork_pid_shortcut_clears_in_use_flag() -> None:
    """Set ``_in_use=True`` to simulate a mid-operation state at fork
    time, then point ``_creator_pid`` at a different pid so the
    fork-pid shortcut fires. After close(), ``_in_use`` must be
    False so the next method call passes ``_check_in_use``."""
    conn = _make_connection()
    conn._in_use = True
    # Simulate forked-from-parent state — pid mismatch triggers the
    # shortcut branch.
    conn._creator_pid = os.getpid() + 1_000_000

    await conn.close()

    assert conn._closed is True
    assert conn._in_use is False, (
        "fork-pid shortcut must clear _in_use; otherwise the forked "
        "child is permanently locked behind _check_in_use's 'another "
        "operation is in progress' diagnostic"
    )


# Quiet asyncio import lint.
_ = asyncio
