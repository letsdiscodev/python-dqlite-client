"""Pin: ``_drain_idle``'s per-iteration cap clips to the remaining caller
deadline, so one stuck close cannot overshoot the overall budget."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_drain_idle_clips_per_iter_cap_to_remaining_deadline() -> None:
    """A queued conn whose close parks forever: drain wall-clock must be bounded
    by the tight deadline, not by the per-iter cap."""
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._closed = False
    pool._max_size = 1
    pool._size = 1
    pool._close_timeout = 0.5  # per-iter cap = 0.5 × 4 = 2.0s
    pool._pool = asyncio.Queue()
    pool._lock = asyncio.Lock()
    pool._waiters = 0  # type: ignore[attr-defined]
    pool._closed_event = None

    stuck = MagicMock()
    stuck._pool_released = True

    parked = asyncio.Event()
    started = asyncio.Event()

    async def _parked_close() -> None:
        started.set()
        await parked.wait()

    stuck.close = _parked_close
    pool._pool.put_nowait(stuck)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.2  # tight; pre-fix the per-iter cap ran to 2.0s

    start = loop.time()
    await pool._drain_idle(deadline=deadline)
    elapsed = loop.time() - start

    assert elapsed < 0.6, (
        f"_drain_idle overshot the deadline by ~{elapsed - 0.2:.3f}s; "
        f"the per-iter cap should clip to the remaining deadline"
    )

    parked.set()  # release the parked close so the orphan task can clean up
