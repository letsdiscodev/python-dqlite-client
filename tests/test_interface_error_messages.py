"""Pin the exact wording of every ``InterfaceError`` raised from
``DqliteConnection._check_in_use``.

Operators triage on substring matches in error messages; downstream
tooling (SQLAlchemy's ``is_disconnect``, retry decorators, log
parsers) often inspects the wording too. A future cleanup that
rewords any of these branches would silently break parsers.

Existing tests pin the "another operation is in progress" branch
(``test_check_in_use_error_includes_task_identity.py``) and the
"owned by another task" branch (same file). This file pins the
remaining three branches:

  * pool-released
  * cross-loop binding mismatch
  * called from sync context (no running loop)
"""

from __future__ import annotations

import asyncio
import weakref
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


def _make_bound_connection() -> DqliteConnection:
    """A connection in the post-bind state on the running loop, with
    every other ``_check_in_use`` precondition relaxed."""
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
    # A different loop than the running one — cannot construct a real
    # second loop here; a sentinel that compares unequal is enough,
    # because ``_check_in_use`` only does ``is`` identity comparison.
    sentinel = MagicMock(spec=asyncio.AbstractEventLoop)
    conn._bound_loop_ref = weakref.ref(sentinel)
    with pytest.raises(InterfaceError, match="bound to a different event loop"):
        conn._check_in_use()


def test_called_from_sync_context_branch_message_substring() -> None:
    """Outside a running loop, ``_check_in_use`` raises with a clear
    message rather than crashing on ``get_running_loop``'s
    RuntimeError. Run as a SYNC test so the surrounding code does
    not have a running loop."""
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


# ``weakref`` is used by the helpers above to set up
# ``_bound_loop_ref`` in the bound-state fixture.
