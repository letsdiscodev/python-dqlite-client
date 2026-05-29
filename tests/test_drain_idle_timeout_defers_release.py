"""_drain_idle's TimeoutError arm defers the slot release until the orphan close() completes.

Releasing immediately lets a subsequent acquire() dial a new conn while the orphan still runs,
transiently breaching max_size — fatal against a hard cluster-side connection cap.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


async def test_drain_idle_timeout_does_not_release_slot_until_drain_completes() -> None:
    """A queued conn whose close() blocks past the cap keeps _size elevated until it finishes."""
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._close_timeout = 0.01
    pool._closed = False
    pool._closed_event = None
    pool._lock = asyncio.Lock()
    pool._size = 1
    pool._max_size = 5
    pool._pool = asyncio.Queue()

    blocker = asyncio.Event()
    close_started = asyncio.Event()
    close_completed = asyncio.Event()

    class _StubConn:
        _address = "stub:9001"
        _pool_released = True

        async def close(self) -> None:
            close_started.set()
            try:
                await blocker.wait()
            finally:
                close_completed.set()

    stub = _StubConn()
    await pool._pool.put(stub)  # type: ignore[arg-type]

    drain_task = asyncio.create_task(pool._drain_idle())
    await close_started.wait()
    await drain_task

    # Slot still reserved because the orphan close has not completed yet.
    assert pool._size == 1, (
        "TimeoutError on per-iteration drain must defer the slot "
        f"release until the orphan close finishes; observed _size={pool._size}"
    )

    # Let the orphan drain finish so the deferred release fires.
    blocker.set()
    await close_completed.wait()
    for _ in range(50):
        if pool._size == 0:
            break
        await asyncio.sleep(0.01)

    assert pool._size == 0, (
        f"deferred slot release must fire once the orphan close finishes; "
        f"observed _size={pool._size}"
    )
