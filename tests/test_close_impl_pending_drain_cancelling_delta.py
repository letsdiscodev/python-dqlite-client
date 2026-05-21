"""Pin: ``DqliteConnection._close_impl``'s pending-drain re-snapshot
loop uses the ``Task.cancelling()``-delta dance to distinguish a
FRESH outer cancel landing during ``await pending`` from a
third-party cancel that propagated through the pending task itself.

The prior shape used ``contextlib.suppress(Exception,
asyncio.CancelledError)``, which silently swallowed ANY
CancelledError — including a fresh outer cancel from a TaskGroup
sibling failure or a manual ``task.cancel()`` + ``await task``
idiom. The caller's ``except CancelledError`` arm never fired; the
structured-concurrency contract was broken.

Symmetric with the ``_connect_impl`` sibling discipline (already
pinned at ``test_connect_impl_pending_drain_resnapshot_loop``).
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_propagates_fresh_outer_cancel_during_drain() -> None:
    """A fresh outer ``task.cancel()`` landing while ``_close_impl``
    is awaiting the pending drain must propagate as ``CancelledError``
    to the awaiter. The prior ``suppress(CancelledError)`` swallowed
    it, leaving the caller blind to a structured-concurrency cancel.
    """
    loop = asyncio.get_running_loop()

    # Long-running drain so we have time to schedule the outer cancel.
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
    # Schedule the fresh outer cancel to fire while close_impl is
    # suspended in ``await pending``.
    loop.call_later(0.01, closer_task.cancel)
    try:
        outcome = await closer_task
    except asyncio.CancelledError:
        outcome = "raised"
    assert outcome == "raised", (
        "fresh outer cancel during _close_impl's pending-drain await "
        "must propagate as CancelledError, not be silently swallowed"
    )
    # Pending drain task was cancelled in the process.
    assert pending.cancelled() or pending.done()


@pytest.mark.asyncio
async def test_close_impl_consumes_third_party_cancel_on_pending() -> None:
    """When the pending drain task is cancelled by a THIRD party
    (not the current task's own ``task.cancel()``), the resulting
    ``CancelledError`` on ``await pending`` must be consumed so the
    close can proceed — OUR ``cancelling()`` counter is unchanged.
    """
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

    # Third party cancels the pending drain directly; the current
    # task's cancelling-counter is NOT touched.
    loop.call_soon(pending.cancel)
    # _close_impl must complete cleanly — the third-party cancel was
    # consumed inside the resnapshot loop.
    await conn._close_impl()
    assert pending.cancelled()
    assert conn._pending_drain is None


@pytest.mark.asyncio
async def test_close_impl_under_asyncio_timeout_surfaces_timeout_error() -> None:
    """When ``_close_impl`` is wrapped by ``asyncio.timeout(epsilon)``
    and the drain takes longer, the parent must observe
    ``TimeoutError`` via the cancelling-counter detection in
    ``asyncio.timeout.__aexit__`` — this remains true even after the
    cancelling-delta migration because we do NOT call
    ``Task.uncancel()`` on the consume path.
    """
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
