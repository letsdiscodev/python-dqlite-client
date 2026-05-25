"""Pin: ``ConnectionPool._drain_pending_under_shield`` balances the
``Task.cancelling()`` count after absorbing a CancelledError, per
PEP 654 / asyncio cancellation bookkeeping.

Without ``current_task().uncancel()`` after the absorb, outer
``asyncio.TaskGroup`` / custom supervisors that gate on
``Task.cancelling()`` see an inflated count — every absorbed cancel
adds one but the helper never balances. ``asyncio.timeout``'s
internal "did we cancel?" probe also relies on the count being
accurate.

The absorb itself is intentional (cleanup must continue across a
fresh cancel); only the bookkeeping was missing.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


async def test_drain_pending_under_shield_uncancels_after_absorb() -> None:
    pool = ConnectionPool.__new__(ConnectionPool)

    # Build a pending-drain task that the helper will await under
    # shield.
    pending_blocker = asyncio.Event()

    async def slow_pending() -> None:
        await pending_blocker.wait()

    pending_task = asyncio.create_task(slow_pending())

    class _StubConn:
        _pending_drain = pending_task
        _address = "stub:9001"

    stub = _StubConn()

    initial_cancelling = 0

    async def run_helper_under_cancel() -> None:
        nonlocal initial_cancelling
        current = asyncio.current_task()
        assert current is not None
        initial_cancelling = current.cancelling()
        # Cancel ourselves so the await inside the helper raises
        # CancelledError, which the helper absorbs.
        current.cancel()
        await pool._drain_pending_under_shield(stub)  # type: ignore[arg-type]
        # After the helper returns, the cancel was absorbed and the
        # cancelling count must have been balanced. Without uncancel,
        # the count stays inflated by one.
        post = current.cancelling()
        assert post == initial_cancelling, (
            f"_drain_pending_under_shield must balance the cancelling() "
            f"count after absorbing a CancelledError; "
            f"initial={initial_cancelling}, post={post}"
        )

    task = asyncio.create_task(run_helper_under_cancel())
    await asyncio.wait_for(task, timeout=2.0)
    # Tear down the pending drain task.
    pending_blocker.set()
    pending_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await pending_task


async def test_drain_pending_under_shield_no_pending_is_noop() -> None:
    """When ``_pending_drain`` is None the helper returns without
    touching the cancel state.
    """
    pool = ConnectionPool.__new__(ConnectionPool)

    class _StubConn:
        _pending_drain = None
        _address = "stub:9001"

    stub = _StubConn()
    # The helper should return cleanly without raising.
    await pool._drain_pending_under_shield(stub)  # type: ignore[arg-type]
