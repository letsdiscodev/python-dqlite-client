"""Pin: ``_invalidate`` does NOT clear ``_in_use``.

Clearing it out-of-band would let a second task enter ``_run_protocol``'s
critical section while the original claimant is still mid-``await``. The
in-flight task clears the flag itself in its own ``finally``.
"""

from __future__ import annotations

import weakref

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_invalidate_preserves_in_use_so_concurrent_claimant_is_rejected() -> None:
    """A concurrent ``_invalidate`` must not clear Task A's ``_in_use``;
    a second task hitting ``_check_in_use`` must still raise."""
    import asyncio

    loop = asyncio.get_running_loop()

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._in_use = True  # Task A's mid-_run_protocol claim
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 5
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop_ref = weakref.ref(loop)
    conn._close_timeout = 5.0
    conn._pending_drain = None
    conn._address = "127.0.0.1:1"
    import os

    conn._creator_pid = os.getpid()
    conn._pool_released = False

    # Out-of-band _invalidate (e.g. from call_soon_threadsafe).
    conn._invalidate(RuntimeError("simulated heartbeat invalidation"))

    assert conn._in_use is True, (
        "_invalidate must not clear _in_use; otherwise a concurrent "
        "task can enter _run_protocol's critical section while the "
        "original claimant is still mid-await"
    )

    with pytest.raises(InterfaceError, match="another operation is in progress"):
        conn._check_in_use()


@pytest.mark.asyncio
async def test_invalidate_with_no_active_claimant_leaves_in_use_false() -> None:
    """With ``_in_use=False`` going in, ``_invalidate`` must not flip it to True."""
    import asyncio

    loop = asyncio.get_running_loop()

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._in_use = False
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 5
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop_ref = weakref.ref(loop)
    conn._close_timeout = 5.0
    conn._pending_drain = None
    conn._address = "127.0.0.1:1"
    import os

    conn._creator_pid = os.getpid()
    conn._pool_released = False

    conn._invalidate(RuntimeError("simulated"))

    assert conn._in_use is False
