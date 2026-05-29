"""Pool reset must tolerate InterfaceError from _check_in_use (a conn
whose _tx_owner is a still-live other task): treat as a clean drop, not
a crash that leaks the _size slot.
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
    """A conn with a stale tx-owner from another live task is dropped
    (return False), not crashed with a leaked _size slot."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)

    conn = DqliteConnection("localhost:9001")
    # is_connected=True forces the ROLLBACK branch, not the early return.
    with patch.object(DqliteConnection, "is_connected", new=True):
        conn._in_transaction = True
        conn._tx_owner = asyncio.current_task()

        async def _release_from_a_different_task() -> bool:
            # Reset from a new task so _check_in_use sees the original
            # task as tx owner and raises InterfaceError.
            return await pool._reset_connection(conn)

        sibling = asyncio.create_task(_release_from_a_different_task())
        result = await sibling

    assert result is False
