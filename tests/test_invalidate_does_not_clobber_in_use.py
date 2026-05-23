"""Pin: ``_invalidate`` does NOT clear ``_in_use``.

The flag's contract is "owned by the task that claimed it between
claim-site and clear-site". Clearing it from an out-of-band path
(``call_soon_threadsafe`` from the dbapi sync-timeout arm,
heartbeat invalidations, ``transaction()`` rollback-failure arm)
would let a second task enter a ``_run_protocol``-protected
critical section while the original claimant is still mid-``await``.

The in-flight task observes the nulled ``_protocol`` at its next
read / write and runs its own ``finally`` to clear ``_in_use`` —
the post-fix shape never lets a concurrent caller see a
spuriously-cleared flag mid-await.
"""

from __future__ import annotations

import weakref

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import InterfaceError


@pytest.mark.asyncio
async def test_invalidate_preserves_in_use_so_concurrent_claimant_is_rejected() -> None:
    """Task A holds ``_in_use=True`` (the in-flight claimant). A
    concurrent ``_invalidate`` lands. The flag must NOT be cleared
    here — a second task hitting ``_check_in_use`` must still see
    the in-progress claim and raise.
    """
    import asyncio

    loop = asyncio.get_running_loop()

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._in_use = True  # simulate Task A's mid-_run_protocol claim
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

    # Load-bearing: the flag was NOT cleared. A second task hitting
    # ``_check_in_use`` must observe the still-claimed state.
    assert conn._in_use is True, (
        "_invalidate must not clear _in_use; otherwise a concurrent "
        "task can enter _run_protocol's critical section while the "
        "original claimant is still mid-await"
    )

    # ``_check_in_use`` rejects with InterfaceError.
    with pytest.raises(InterfaceError, match="another operation is in progress"):
        conn._check_in_use()


@pytest.mark.asyncio
async def test_invalidate_with_no_active_claimant_leaves_in_use_false() -> None:
    """When no task holds the flag (``_in_use=False`` going in),
    ``_invalidate`` must not flip it to True. The fix is a delete, not
    a swap.
    """
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

    # The flag was untouched — and is still safely false.
    assert conn._in_use is False
