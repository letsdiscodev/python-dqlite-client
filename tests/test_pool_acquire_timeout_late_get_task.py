"""Pin: ``acquire()``'s capacity-wait timeout demux must not silently
discard a connection that resolved ``get_task`` during the post-wait
``await closed_task`` yield.

When ``asyncio.wait`` returns the timeout (``done == set()``) and the
post-wait code cancels and awaits ``closed_task``, that await yields to
the scheduler. A sibling ``_pool.put_nowait(conn)`` running at that
yield resolves the still-pending ``get_task``. The original code's
demux test ``if get_task in done`` uses the *snapshot* taken before the
yield, so it incorrectly takes the else-arm: cancels (no-op on a done
task), ``await get_task`` returns the connection, ``continue`` discards
it. The reservation slot is never released because ``_release`` only
fires for connections that flow back through the user's context
manager. The pool permanently loses one slot of capacity per
occurrence.

The fix replaces the snapshot-membership check with a live state check
``get_task.done() and not get_task.cancelled() and get_task.exception()
is None`` and routes a winning conn through the same put-back-or-release
helper used by the existing ``except BaseException`` arm.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol = MagicMock()
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False
        self.close_called = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        self._protocol = None  # type: ignore[assignment]


def _make_pool() -> ConnectionPool:
    async def _connect(**_: Any) -> _FakeConn:
        return _FakeConn()

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _connect
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=0.1,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_acquire_timeout_race_does_not_discard_late_winning_get_task() -> None:
    """``asyncio.wait`` times out (``done == set()``) but a sibling
    ``put_nowait`` resolves ``get_task`` during the post-wait
    ``await closed_task`` yield. The conn must be put back on the
    queue (or routed to the user), not silently discarded by the
    stale-snapshot demux.

    Setup: simulate "pool already at max_size" by directly setting
    ``_size = 1`` (no real held connection). When ``acquire()`` enters
    the capacity-wait branch, our patched ``asyncio.wait`` drops a
    phantom connection into the queue synchronously before returning
    a timeout-shaped result (done=empty, both tasks still pending).
    The next loop iteration delivered to ``await closed_task`` then
    runs ``get_task.__step`` before our coroutine resumes, so
    ``get_task`` consumes the phantom and becomes done. The buggy
    post-wait demux at ``pool.py`` ``if get_task in done`` sees the
    stale empty snapshot, takes the else-arm, and discards the
    phantom on ``continue``. The fix re-checks ``get_task.done()``
    live and routes the conn through put-back-or-release.
    """
    pool = _make_pool()

    # Pretend max_size is reached so ``acquire()`` can't reserve and
    # enters the capacity-wait branch on its first iteration. Avoids
    # the asynccontextmanager-finalization noise from holding a real
    # acquire's slot across a bare ``__aenter__()``.
    pool._size = 1

    phantom = _FakeConn(name="phantom")
    original_put_nowait = pool._pool.put_nowait

    import dqliteclient.pool as pool_mod

    real_wait = asyncio.wait
    call_count = 0

    async def fake_wait(
        tasks: Any, *, timeout: Any = None, return_when: Any = None
    ) -> tuple[set[Any], set[Any]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Drop the phantom into the queue while ``get_task`` has
            # not yet had its first ``__step`` run (we are inside
            # ``await asyncio.wait`` synchronously — the loop hasn't
            # iterated since ``create_task``). When the post-wait code
            # subsequently yields on ``await closed_task``, the loop
            # runs ``get_task.__step`` first and ``get_task`` consumes
            # ``phantom``, becoming done with a real result, before
            # our coroutine resumes. The post-wait demux's
            # ``if get_task in done`` then sees the stale empty
            # snapshot and routes ``get_task`` into the
            # cancel-and-discard arm.
            original_put_nowait(phantom)  # type: ignore[arg-type]
            # Return timeout: done=empty, both tasks still pending
            # from the snapshot's perspective.
            return set(), set(tasks)
        # Subsequent calls: defer to the real wait so the deadline
        # actually consumes time and the loop terminates promptly.
        return await real_wait(tasks, timeout=timeout, return_when=return_when)

    pool_mod.asyncio.wait = fake_wait  # type: ignore[attr-defined]
    received: object | None = None
    try:
        # With the fix: the live-state recheck after ``await
        # closed_task`` finds get_task done with phantom and routes it
        # to the user (no timeout). Without the fix: stale snapshot
        # demux drops phantom on the floor; subsequent iterations time
        # out with an empty queue.
        async with pool.acquire() as conn:
            received = conn
    finally:
        pool_mod.asyncio.wait = real_wait  # type: ignore[attr-defined]

    # The phantom that ``put_nowait`` deposited during the
    # capacity-wait race must reach the user (or round-trip back to
    # the queue), never be silently discarded by the stale ``done``
    # snapshot demux.
    assert received is phantom, (
        f"acquire returned {received!r}, not the phantom that was put "
        "into the queue during the capacity-wait race — the post-wait "
        "demux's stale 'done' snapshot dropped phantom on the floor"
    )

    # Reset _size to the value used to simulate at-capacity so close()
    # does not hit the underflow guard. ``_release`` already
    # decremented _size by routing through ``_release_reservation`` on
    # __aexit__.
    pool._size = 0
    await pool.close()


@pytest.mark.asyncio
async def test_put_back_or_release_late_winner_queuefull_falls_back_to_close() -> None:
    """If the queue is full when the late-winner helper tries to put,
    it must close the conn and release the reservation rather than
    silently leak it.

    The QueueFull branch represents an "impossible" reservation-vs-
    capacity violation; the helper must handle it without dropping
    the conn on the floor or skipping the ``_size`` decrement that
    wakes sibling acquirers.
    """
    pool = _make_pool()

    # Pre-fill the bounded queue (max_size=1) so the helper's
    # put_nowait immediately raises QueueFull.
    pool._size = 1
    occupant = _FakeConn(name="occupant")
    pool._pool.put_nowait(occupant)  # type: ignore[arg-type]
    assert pool._pool.full()

    late_winner = _FakeConn(name="late_winner")
    await pool._put_back_or_release_late_winner(late_winner)  # type: ignore[arg-type]

    # The late_winner must have been close()'d (because put_nowait
    # raised QueueFull, the helper falls back to close + release).
    assert late_winner.close_called is True

    # The reservation must have been released (size -= 1) so a
    # sibling acquirer can replace the slot.
    assert pool._size == 0

    # Cleanup: drain the occupant from the queue.
    queued = pool._pool.get_nowait()
    assert queued is occupant
    await pool.close()
