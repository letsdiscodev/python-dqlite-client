"""``pool._drain_idle`` survives an outer cancel landing mid-``conn.close()``.

Without per-connection shielding, an ``asyncio.timeout(pool.close())``
that fires while one ``conn.close()`` is blocked on ``wait_closed``
propagates ``CancelledError`` up through the drain loop and leaves
every subsequent queued connection orphaned — their reader tasks and
transports leak until GC at interpreter exit.
"""

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

    # Build the pool skeleton directly, bypassing `initialize()`.
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._pool = asyncio.Queue()
    for f in fakes:
        pool._pool.put_nowait(f)  # type: ignore[arg-type]
    pool._size = len(fakes)
    pool._max_size = len(fakes)
    pool._lock = asyncio.Lock()
    pool._closed = False
    pool._closed_event = None

    # Race an outer cancel against the drain.
    drain_task = asyncio.create_task(pool._drain_idle())
    await asyncio.sleep(0.05)  # let the drain start the first close
    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    # The drain loop bails on cancel, but the shielded per-conn close
    # keeps running. Yield long enough for it to settle, then assert
    # every STARTED close eventually COMPLETED — without the shield,
    # cancel propagates into ``wait_closed`` and the close never
    # finishes.
    await asyncio.sleep(0.2)
    assert set(started).issubset({0, 1, 2})
    assert started  # at least one close was started before the cancel
    assert set(started) == set(completed), (
        f"some close() started but did not complete: started={started} completed={completed}"
    )
