"""Pool reset must tolerate InterfaceError from _check_in_use.

If a connection's `_in_transaction=True` and `_tx_owner` points at a
still-live task, `conn.execute("ROLLBACK")` from `_reset_connection`
goes through `_run_protocol → _check_in_use` which raises
`InterfaceError("owned by another task")`. That InterfaceError must be
classified as a clean drop-the-connection event, NOT crash the pool's
release path and leak the `_size` slot.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import _POOL_CLEANUP_EXCEPTIONS, ConnectionPool


def test_pool_cleanup_exceptions_include_interface_error() -> None:
    """Pin that InterfaceError is in the cleanup-tolerated set."""
    assert InterfaceError in _POOL_CLEANUP_EXCEPTIONS


@pytest.mark.asyncio
async def test_reset_connection_tolerates_interface_error_from_check_in_use() -> None:
    """A connection released to the pool with a stale tx-owner from
    another live task must not crash the pool — it should be treated
    as a non-reusable connection and dropped. Without this fix the
    InterfaceError from _check_in_use propagates out of
    _reset_connection and the pool's release path crashes, leaking
    the connection's `_size` slot."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)

    # Build a connection that "looks" like it belongs to a different
    # task that is still alive.
    conn = DqliteConnection("localhost:9001")
    # Force is_connected to True so _reset_connection takes the ROLLBACK
    # branch instead of the early "not connected" return.
    with patch.object(DqliteConnection, "is_connected", new=True):
        conn._in_transaction = True
        # Use the current task as the (different) owner from the pool's
        # release task. Real-world callers manage this via the
        # transaction() ctxmgr; here we set the flag directly to
        # exercise the defensive path.
        conn._tx_owner = asyncio.current_task()

        async def _release_from_a_different_task() -> bool:
            # Run the reset from a brand-new task so _check_in_use sees
            # the original (current) task as the tx owner and raises
            # InterfaceError.
            return await pool._reset_connection(conn)

        sibling = asyncio.create_task(_release_from_a_different_task())
        result = await sibling

    # _reset_connection should return False (drop the connection) without
    # propagating InterfaceError.
    assert result is False
