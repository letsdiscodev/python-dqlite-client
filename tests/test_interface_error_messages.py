"""Pin the wording of ``InterfaceError`` messages from
``_check_in_use`` (pool-released, cross-loop, sync-context branches);
downstream tooling triages on these substrings.
"""

from __future__ import annotations

import asyncio
import weakref
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def _make_bound_connection() -> DqliteConnection:
    """Connection in post-bind state with other ``_check_in_use`` preconditions relaxed."""
    import os as _os

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pool_released = False
    conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._creator_pid = _os.getpid()
    return conn


@pytest.mark.asyncio
async def test_pool_released_branch_message_substring() -> None:
    conn = _make_bound_connection()
    conn._pool_released = True
    with pytest.raises(InterfaceError, match="returned to the pool"):
        conn._check_in_use()


@pytest.mark.asyncio
async def test_cross_loop_branch_message_substring() -> None:
    conn = _make_bound_connection()
    # ``_check_in_use`` only does ``is`` comparison, so any unequal sentinel works.
    sentinel = MagicMock(spec=asyncio.AbstractEventLoop)
    conn._bound_loop_ref = weakref.ref(sentinel)
    with pytest.raises(InterfaceError, match="bound to a different event loop"):
        conn._check_in_use()


def test_called_from_sync_context_branch_message_substring() -> None:
    """Sync test (no running loop): ``_check_in_use`` raises a clear
    message rather than ``get_running_loop``'s RuntimeError."""
    import os as _os

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pool_released = False
    conn._bound_loop_ref = None
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._creator_pid = _os.getpid()
    with pytest.raises(InterfaceError, match="from within an async context"):
        conn._check_in_use()
