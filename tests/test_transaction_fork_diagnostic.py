"""Pin: ``DqliteConnection.transaction()`` calls ``_check_in_use`` at
the top so a forked child sees the clear "used after fork" diagnostic
instead of the misleading "owned by another task" branch (which would
render the parent's task repr in the error message — confusing
diagnostic with a non-actionable owner reference).

The transaction context manager previously skipped past
``_check_in_use`` straight into its own nested-tx / sibling-task
checks. After fork, ``_in_transaction`` may be True (parent had a tx
in flight) and ``_tx_owner`` is the parent's Task object — neither is
meaningful in the child, but the "owned by another task" branch fires
and points users at "use a separate connection from the pool" instead
of telling them to reconstruct the connection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_transaction_after_fork_raises_fork_diagnostic_not_cross_task() -> None:
    """A forked child entering ``transaction()`` must see the
    "used after fork" message even if the parent had a transaction
    in flight at fork time. Without ``_check_in_use`` at the top, the
    nested-task branch fires and surfaces the parent's Task repr."""
    conn = DqliteConnection("127.0.0.1:9999")
    # Stage a parent-side transaction in flight: _in_transaction True
    # and _tx_owner pointing at a fake "parent task." After fork the
    # child inherits both fields.
    conn._in_transaction = True
    conn._tx_owner = MagicMock(spec=asyncio.Task)
    conn._bound_loop = asyncio.get_running_loop()
    fake_parent_pid = conn._creator_pid + 1
    conn._creator_pid = fake_parent_pid

    # Patch the module-level pid cache so the misuse guard observes a
    # fresh-process pid different from ``_creator_pid``. Cycle 21 moved
    # the hot-path pid check from ``os.getpid()`` to a cached module
    # attribute updated via ``os.register_at_fork``; patching
    # ``os.getpid`` would be dead code.
    with (
        patch("dqliteclient.connection._current_pid", fake_parent_pid + 1),
        pytest.raises(InterfaceError, match="fork") as excinfo,
    ):
        async with conn.transaction():
            pass

    # The diagnostic must be the fork message, not the cross-task one.
    msg = str(excinfo.value)
    assert "fork" in msg
    assert "owned by another task" not in msg
