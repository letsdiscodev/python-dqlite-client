"""Pin: ``ConnectionPool._drain_idle``'s per-iteration cap is
clipped to the remaining caller-supplied deadline so an
intermediate stuck close cannot overshoot the documented overall
budget by ~3× close_timeout.

Pre-fix the deadline gate only fired BETWEEN iterations; the
per-iteration ``wait_for`` was bounded by
``close_timeout × _DRAIN_PER_CONN_CAP_MULTIPLIER`` (typically 4×)
regardless of how much budget remained. A single stuck close could
extend the drain by ``(multiplier - 1) × close_timeout`` past the
deadline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_drain_idle_clips_per_iter_cap_to_remaining_deadline() -> None:
    """Stage one queued conn whose close parks indefinitely; call
    _drain_idle with a tight deadline. The total wall-clock must
    be bounded by the deadline, not by the per-iter cap."""
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
    # Tight deadline: 0.2s. Pre-fix the per-iter cap would extend
    # the drain to 2.0s.
    deadline = loop.time() + 0.2

    start = loop.time()
    await pool._drain_idle(deadline=deadline)
    elapsed = loop.time() - start

    # Post-fix: bounded by the deadline (with small grace). Pre-fix
    # the wait_for runs the full 2.0s per-iter cap.
    assert elapsed < 0.6, (
        f"_drain_idle overshot the deadline by ~{elapsed - 0.2:.3f}s; "
        f"the per-iter cap should clip to the remaining deadline"
    )

    # Release the parked close so the orphan task can clean up.
    parked.set()
