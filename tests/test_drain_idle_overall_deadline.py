"""Pin: ``ConnectionPool._drain_idle`` honours an overall ``deadline``
budget so the user-supplied ``pool.timeout`` is not silently violated
under leader-flip / dead-peer cascade.

Without the deadline parameter the per-iteration cap could compound
across ``max_size`` queued connections — a pathologically slow
``close()`` could blow past the documented ``pool.timeout`` by
``max_size × close_timeout × cap_multiplier`` seconds before
``_create_connection`` ran. The surfaced "Timed out creating a fresh
connection" message then mis-attributed the phase.

Pin the behaviour: when ``loop.time() >= deadline`` between
iterations, the drain returns and the remaining idle queue is left
to the after-cancel sweep / next acquire.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


class _SlowCloseConn:
    """Stand-in for ``DqliteConnection`` whose ``close()`` parks until
    cancelled. The ``_drain_idle`` per-iteration cap fires on each
    such conn — without the overall deadline we'd hit the cap N times
    in sequence."""

    def __init__(self) -> None:
        self._pool_released = True
        self._protocol = MagicMock()
        self.is_connected = True
        self._address = "slow:9001"

    async def close(self, *_args: object, **_kwargs: object) -> None:
        # Park forever — the per-iteration ``wait_for`` cap is what
        # eventually cancels us. Without the OVERALL deadline, every
        # queued slow conn pays the cap separately.
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
    # Fill the queue with four slow conns. Without the deadline gate
    # each pays the per-iteration cap; with it, only one (or zero)
    # iterations run inside the budget.
    for _ in range(4):
        pool._pool.put_nowait(_SlowCloseConn())  # type: ignore[arg-type]
    pool._size = 4

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.06  # tight overall budget

    start = loop.time()
    # Should return promptly via the between-iteration check; the per-
    # iteration cap is _close_timeout * multiplier = 0.05 * 4 = 0.2s,
    # so a single iteration's wait_for might still take that long, but
    # the OVERALL drain must not consume 4 × 0.2 = 0.8s.
    await pool._drain_idle(deadline=deadline)
    elapsed = loop.time() - start

    # 0.5s gives generous slack for the single-iteration cap to fire
    # while still proving the overall bound is honoured (no compounded
    # 0.8s, 1.6s, etc.).
    assert elapsed < 0.5, f"_drain_idle exceeded the overall deadline: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_drain_idle_without_deadline_keeps_old_unbounded_behaviour() -> None:
    """Negative pin: when ``deadline=None`` the loop processes every
    queued conn — the new parameter is opt-in."""
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

    await pool._drain_idle()  # no deadline
    assert sorted(closed) == [0, 1, 2]
