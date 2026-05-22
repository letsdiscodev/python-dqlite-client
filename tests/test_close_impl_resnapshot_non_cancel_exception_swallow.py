"""Pin: ``DqliteConnection._close_impl``'s pending-drain re-snapshot
loop swallows non-cancel ``Exception`` raises (e.g.,
``RuntimeError("Transport is closed")`` from a half-torn writer's
``wait_closed``) and completes the close cleanly without leaking
the exception or orphaning the pending task.

The sibling cancel arm at connection.py:1925-1933 is pinned by
``tests/test_close_impl_pending_drain_cancelling_delta.py``. The
non-cancel ``except Exception`` arm at connection.py:1934-1940
was previously uncovered — a regression that narrowed the catch or
removed the swallow could let transport noise bubble up and
abort the close path.
"""

from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


async def test_close_impl_swallows_non_cancel_exception_from_pending_drain() -> None:
    """Construct a connection whose ``_pending_drain`` raises a
    non-cancel ``Exception`` on await. ``_close_impl`` must absorb
    it via the bare ``except Exception:`` arm, clear
    ``_pending_drain``, and complete cleanly. No
    "Task exception was never retrieved" warning at GC.
    """
    loop = asyncio.get_running_loop()

    async def _raises_non_cancel() -> None:
        raise RuntimeError("Transport is closed")

    pending = loop.create_task(_raises_non_cancel())

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = pending
    conn._protocol = None
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop_ref = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # close_impl must complete without raising — the resnapshot
        # loop's Exception arm swallows the non-cancel raise.
        await conn._close_impl()
        # Pending task is observed (done()), slot is cleared.
        assert pending.done()
        assert conn._pending_drain is None
        # Force GC to surface any orphaned-task warnings.
        gc.collect()

    bad = [
        str(w.message)
        for w in caught
        if "Task exception was never retrieved" in str(w.message)
        or ("destroyed" in str(w.message).lower() and "pending" in str(w.message).lower())
    ]
    assert bad == [], (
        f"pending-drain Exception arm orphaned the task; got {bad!r}. "
        "The resnapshot loop must await the pending task directly so its "
        "exception is observed."
    )
