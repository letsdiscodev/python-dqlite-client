"""ConnectionPool._drain_idle honours an overall deadline so pool.timeout isn't silently violated.

Without it the per-iteration cap compounds across max_size queued conns
(max_size x close_timeout x cap_multiplier), blowing past the documented timeout. On deadline the
drain returns and leaves the remaining idle queue to the after-cancel sweep / next acquire.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


class _SlowCloseConn:
    """DqliteConnection stand-in whose close() parks until the per-iteration cap cancels it."""

    def __init__(self) -> None:
        self._pool_released = True
        self._protocol = MagicMock()
        self.is_connected = True
        self._address = "slow:9001"

    async def close(self, *_args: object, **_kwargs: object) -> None:
        # Park forever; the per-iteration wait_for cap cancels us.
        await asyncio.sleep(3600)


def _make_pool() -> ConnectionPool:
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._pool = asyncio.Queue(maxsize=4)
    pool._size = 0
    pool._lock = asyncio.Lock()
    pool._closed = False
    pool._closed_event = None
    pool._close_done = None
    pool._creator_pid = -1  # sentinel — never matches a real pid
    pool._close_timeout = 0.05
    pool._max_size = 4
    pool._cluster = None  # type: ignore[assignment]
    pool._timeout = 1.0
    pool._addresses = []
    return pool


@pytest.mark.asyncio
async def test_drain_idle_returns_when_deadline_passed_between_iterations() -> None:
    pool = _make_pool()
    for _ in range(4):
        pool._pool.put_nowait(_SlowCloseConn())  # type: ignore[arg-type]
    pool._size = 4

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.06  # tight overall budget

    start = loop.time()
    # Single-iteration cap is 0.2s; the overall drain must not compound to 4 x 0.2 = 0.8s.
    await pool._drain_idle(deadline=deadline)
    elapsed = loop.time() - start

    assert elapsed < 0.5, f"_drain_idle exceeded the overall deadline: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_drain_idle_without_deadline_keeps_old_unbounded_behaviour() -> None:
    """With deadline=None the loop processes every queued conn; the parameter is opt-in."""
    pool = _make_pool()

    closed: list[int] = []

    class _FastCloseConn:
        def __init__(self, idx: int) -> None:
            self._pool_released = True
            self._idx = idx
            self._address = f"node:{idx}"

        async def close(self, *_args: object, **_kwargs: object) -> None:
            closed.append(self._idx)

    for i in range(3):
        pool._pool.put_nowait(_FastCloseConn(i))  # type: ignore[arg-type]
    pool._size = 3

    await pool._drain_idle()
    assert sorted(closed) == [0, 1, 2]
