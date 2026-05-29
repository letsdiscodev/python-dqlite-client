"""``pool._drain_idle`` survives an outer cancel mid-``conn.close()``: without
per-connection shielding the cancel propagates through ``wait_closed`` and
leaves the started close unfinished and later queued conns orphaned."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_per_connection_close_survives_outer_cancel() -> None:
    started: list[int] = []
    completed: list[int] = []

    class FakeConn:
        def __init__(self, conn_id: int) -> None:
            self._id = conn_id
            self._address = f"h{conn_id}:9"

        async def close(self) -> None:
            started.append(self._id)
            await asyncio.sleep(0.1)
            completed.append(self._id)

    fakes = [FakeConn(i) for i in range(3)]

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

    drain_task = asyncio.create_task(pool._drain_idle())
    await asyncio.sleep(0.05)  # let the drain start the first close
    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    # The loop bails on cancel but the shielded close keeps running; let it
    # settle, then assert every started close completed.
    await asyncio.sleep(0.2)
    assert set(started).issubset({0, 1, 2})
    assert started  # at least one close was started before the cancel
    assert set(started) == set(completed), (
        f"some close() started but did not complete: started={started} completed={completed}"
    )
