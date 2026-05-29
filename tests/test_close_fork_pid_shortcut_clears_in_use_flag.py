"""``close()``'s fork-pid shortcut clears ``_in_use`` too; otherwise a forked
worker inheriting ``_in_use=True`` is permanently locked out by ``_check_in_use``."""

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
    """A pid mismatch fires the fork-pid shortcut; close() must clear ``_in_use``."""
    conn = _make_connection()
    conn._in_use = True
    conn._creator_pid = os.getpid() + 1_000_000  # pid mismatch fires the shortcut

    await conn.close()

    assert conn._closed is True
    assert conn._in_use is False, (
        "fork-pid shortcut must clear _in_use; otherwise the forked "
        "child is permanently locked behind _check_in_use's 'another "
        "operation is in progress' diagnostic"
    )


# Quiet asyncio import lint.
_ = asyncio
