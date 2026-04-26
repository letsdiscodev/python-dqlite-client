"""Pin: ``_release``'s ``await asyncio.shield(pending)`` survives an
outer cancel landing mid-drain.

The sibling test ``test_pool_release_pending_drain_awaited_on_cancel``
pins that the drain runs *before* ``_pool_released`` flips. This test
covers the second axis: when an outer caller cancels the
``_release(conn)`` task while the drain is in flight, the drain task
itself must complete (shield contract) and ``_pool_released`` must
still be set in the finally.

Without the shield, the inner ``await pending`` would receive the
outer cancel, abort the drain mid-flight, and the reader-task could
outlive the connection on shutdown — the very leak the original
``shield`` exists to prevent.

(The outer cancel itself is absorbed by the surrounding
``contextlib.suppress(CancelledError)`` around the post-drain
``_release_reservation`` shield — see the finally block in
``pool._release``. That absorption is intentional: ``_release`` is a
cleanup path and the queue invariants must always hold; the original
cancel survives via the caller's own check on ``Task.cancelling()``
when the caller chose to cancel us.)
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_outer_cancel_during_drain_does_not_abort_drain() -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True

    drain_started = asyncio.Event()
    drain_release = asyncio.Event()
    drain_completed = False

    async def slow_drain() -> None:
        nonlocal drain_completed
        drain_started.set()
        await drain_release.wait()
        # Note: a few extra yields to make sure we run to completion
        # rather than racing with the surrounding cancel.
        for _ in range(3):
            await asyncio.sleep(0)
        drain_completed = True

    conn._pending_drain = asyncio.create_task(slow_drain())

    async def fake_reset(c: DqliteConnection) -> bool:
        return False

    pool._reset_connection = fake_reset  # type: ignore[assignment]

    release_reservation_ran = False

    async def stub_release_reservation() -> None:
        nonlocal release_reservation_ran
        release_reservation_ran = True

    pool._release_reservation = stub_release_reservation

    async def fake_close() -> None:
        return

    with patch.object(conn, "close", new=fake_close):
        release_task = asyncio.create_task(pool._release(conn))
        await drain_started.wait()
        # Cancel the outer release task while the drain is parked.
        release_task.cancel()
        # Give the cancel one loop turn to land.
        await asyncio.sleep(0)
        # Now release the drain so it can complete despite the cancel.
        drain_release.set()
        # ``_release`` swallows the cancel via its
        # ``contextlib.suppress(CancelledError)`` around
        # ``_release_reservation`` — verify completion either way.
        with contextlib.suppress(asyncio.CancelledError):
            await release_task
        # The shield's ``contextlib.suppress(BaseException)`` returned
        # control to ``_release`` immediately on cancel; the underlying
        # pending drain task is still scheduled and must complete on
        # its own. Drain the remaining loop turns so we can observe it.
        for _ in range(10):
            if conn._pending_drain.done():
                break
            await asyncio.sleep(0)

    # The drain itself must have run to completion (shield contract).
    assert conn._pending_drain.done()
    assert drain_completed
    # The finally clause still flipped the flag.
    assert conn._pool_released is True
    # ``_release_reservation`` is wrapped in shield + suppress, so it
    # ran despite the cancel.
    assert release_reservation_ran is True
