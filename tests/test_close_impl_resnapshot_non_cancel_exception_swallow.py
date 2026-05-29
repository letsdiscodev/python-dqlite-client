"""The pending-drain re-snapshot loop swallows non-cancel ``Exception`` raises (e.g.
``RuntimeError("Transport is closed")`` from a half-torn writer's ``wait_closed``) and
completes the close without leaking the exception or orphaning the task."""

from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


async def test_close_impl_swallows_non_cancel_exception_from_pending_drain() -> None:
    """A ``_pending_drain`` that raises a non-cancel Exception must be absorbed, the slot
    cleared, and the close completed with no "Task exception was never retrieved" warning."""
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
        await conn._close_impl()
        assert pending.done()
        assert conn._pending_drain is None
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
