"""``transaction()`` calls ``_check_in_use`` first so a forked child sees the "used after fork"
diagnostic, not the misleading "owned by another task" branch (which renders the parent's task)."""

from __future__ import annotations

import asyncio
import weakref
from unittest.mock import MagicMock, patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_transaction_after_fork_raises_fork_diagnostic_not_cross_task() -> None:
    """A forked child sees "used after fork" even if the parent had a tx in flight at fork time."""
    conn = DqliteConnection("127.0.0.1:9999")
    # Stage a parent-side tx in flight; the child inherits both fields after fork.
    conn._in_transaction = True
    conn._tx_owner = MagicMock(spec=asyncio.Task)
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
    fake_parent_pid = conn._creator_pid + 1
    conn._creator_pid = fake_parent_pid

    # Make the misuse guard observe a fresh-process pid different from _creator_pid; the
    # hot-path check reads a cached module attribute updated via os.register_at_fork.
    with (
        patch("dqliteclient.connection.os.getpid", return_value=fake_parent_pid + 1),
        pytest.raises(InterfaceError, match="fork") as excinfo,
    ):
        async with conn.transaction():
            pass

    msg = str(excinfo.value)
    assert "fork" in msg
    assert "owned by another task" not in msg
