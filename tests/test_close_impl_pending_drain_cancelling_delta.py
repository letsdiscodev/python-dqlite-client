"""The pending-drain re-snapshot loop uses the ``Task.cancelling()``-delta dance to tell a
FRESH outer cancel during ``await pending`` from a third-party cancel propagated through the
pending task itself; the prior ``suppress(CancelledError)`` swallowed both, breaking the
structured-concurrency contract."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_propagates_fresh_outer_cancel_during_drain() -> None:
    """A fresh outer cancel during ``await pending`` must propagate as CancelledError."""
    loop = asyncio.get_running_loop()

    pending = loop.create_task(asyncio.sleep(0.5))

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

    async def closer() -> str:
        try:
            await conn._close_impl()
        except asyncio.CancelledError:
            return "raised"
        return "swallowed"

    closer_task = loop.create_task(closer())
    loop.call_later(0.01, closer_task.cancel)
    try:
        outcome = await closer_task
    except asyncio.CancelledError:
        outcome = "raised"
    assert outcome == "raised", (
        "fresh outer cancel during _close_impl's pending-drain await "
        "must propagate as CancelledError, not be silently swallowed"
    )
    assert pending.cancelled() or pending.done()


@pytest.mark.asyncio
async def test_close_impl_consumes_third_party_cancel_on_pending() -> None:
    """A third-party cancel on the pending drain (not our own ``task.cancel()``) must be
    consumed so the close proceeds; our ``cancelling()`` counter is unchanged."""
    loop = asyncio.get_running_loop()

    pending = loop.create_task(asyncio.sleep(0.5))

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

    loop.call_soon(pending.cancel)
    await conn._close_impl()
    assert pending.cancelled()
    assert conn._pending_drain is None


@pytest.mark.asyncio
async def test_close_impl_under_asyncio_timeout_surfaces_timeout_error() -> None:
    """``asyncio.timeout(epsilon)`` around a slow drain must surface TimeoutError: the
    cancelling-delta path preserves this by never calling ``Task.uncancel()``."""
    loop = asyncio.get_running_loop()

    pending = loop.create_task(asyncio.sleep(0.5))

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

    saw_timeout = False
    try:
        async with asyncio.timeout(0.01):
            await conn._close_impl()
    except TimeoutError:
        saw_timeout = True
    assert saw_timeout, (
        "asyncio.timeout(epsilon) wrapping _close_impl must surface "
        "TimeoutError via __aexit__'s cancelling-counter detection; "
        "the cancelling-delta dance preserves this contract by not "
        "calling Task.uncancel()."
    )
