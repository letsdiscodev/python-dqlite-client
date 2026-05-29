"""Pin: an outer cancel mid-drain must not leave queued connections orphaned
(never given a close()) — every started close completes and none is leaked."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_drain_does_not_orphan_remaining_queued_connections() -> None:
    started: list[int] = []
    completed: list[int] = []

    class FakeConn:
        def __init__(self, conn_id: int) -> None:
            self._id = conn_id
            self._address = f"h{conn_id}:9"

        async def close(self) -> None:
            started.append(self._id)
            await asyncio.sleep(0.05)
            completed.append(self._id)

    fakes = [FakeConn(i) for i in range(10)]

    pool = ConnectionPool.__new__(ConnectionPool)
    pool._pool = asyncio.Queue()
    for f in fakes:
        pool._pool.put_nowait(f)  # type: ignore[arg-type]
    pool._size = len(fakes)
    pool._max_size = len(fakes)
    pool._lock = asyncio.Lock()
    pool._closed = False
    pool._closed_event = None
    pool._close_timeout = 1.0

    # 0.12s lets the loop start the second conn (0.05s each), then the cancel
    # lands mid-iteration.
    drain_task = asyncio.create_task(pool._drain_idle())
    await asyncio.sleep(0.12)
    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    # Yield long enough for any in-flight shielded closes to land.
    await asyncio.sleep(0.5)

    assert set(started) == set(completed), (
        f"some close() started but did not complete: "
        f"started={sorted(started)} completed={sorted(completed)}"
    )

    # The drain's finally runs _drain_remaining_after_cancel under shield,
    # sweeping any queue entries the cancelled main loop skipped.
    assert pool._pool.empty(), (
        "pool queue should be drained after close cancel — "
        "_drain_remaining_after_cancel must sweep what the main loop did not"
    )
    assert sorted(completed) == list(range(10)), (
        f"every queued connection must have completed close(): completed={sorted(completed)}"
    )
