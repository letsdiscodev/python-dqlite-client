"""Pin: ``_release``'s ``await asyncio.shield(pending)`` survives an outer cancel
landing mid-drain — the drain task completes (shield contract) and _pool_released
is still set in the finally. Without the shield the inner await would take the
cancel, abort the drain, and leak the reader-task on shutdown.
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
        # Extra yields to run to completion rather than race the cancel.
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
        await asyncio.sleep(0)
        # Release the drain so it can complete despite the cancel.
        drain_release.set()
        # _release swallows the cancel via its suppress(CancelledError).
        with contextlib.suppress(asyncio.CancelledError):
            await release_task
        # The drain task is still scheduled; spin until it completes.
        for _ in range(10):
            if conn._pending_drain.done():
                break
            await asyncio.sleep(0)

    assert conn._pending_drain.done()
    assert drain_completed
    assert conn._pool_released is True
    # _release_reservation is shield + suppress wrapped, so it ran despite cancel.
    assert release_reservation_ran is True
