"""_drain_pending_under_shield balances Task.cancelling() after absorbing a CancelledError.

Without uncancel() the count stays inflated, breaking outer TaskGroup/asyncio.timeout supervisors
that gate on it. The absorb itself is intentional; only the bookkeeping was missing.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


async def test_drain_pending_under_shield_uncancels_after_absorb() -> None:
    pool = ConnectionPool.__new__(ConnectionPool)

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
        # Cancel ourselves so the helper's await raises CancelledError, which it absorbs.
        current.cancel()
        await pool._drain_pending_under_shield(stub)  # type: ignore[arg-type]
        post = current.cancelling()
        assert post == initial_cancelling, (
            f"_drain_pending_under_shield must balance the cancelling() "
            f"count after absorbing a CancelledError; "
            f"initial={initial_cancelling}, post={post}"
        )

    task = asyncio.create_task(run_helper_under_cancel())
    await asyncio.wait_for(task, timeout=2.0)
    pending_blocker.set()
    pending_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await pending_task


async def test_drain_pending_under_shield_no_pending_is_noop() -> None:
    """When _pending_drain is None the helper returns without touching the cancel state."""
    pool = ConnectionPool.__new__(ConnectionPool)

    class _StubConn:
        _pending_drain = None
        _address = "stub:9001"

    stub = _StubConn()
    await pool._drain_pending_under_shield(stub)  # type: ignore[arg-type]
