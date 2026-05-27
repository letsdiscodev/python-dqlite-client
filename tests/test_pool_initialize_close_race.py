"""Pool.initialize must not commit connections into a closed pool.

Before the fix, the sequence

    T1: pool.initialize()                T2: pool.close()
      async with self._lock
        self._size += min_size
        await asyncio.gather(...)   ---- suspends on TCP handshake
                                         reads self._pool.qsize() (==0)
                                         self._closed = True
                                         await self._drain_idle()  # empty
                                         close() returns
        gather resolves to N live conns
        for conn in successes:
            await self._pool.put(conn)   # conns now sit in queue of CLOSED pool
        self._initialized = True
        lock released

left ``_min_size`` live connections orphaned in ``self._pool``.
``acquire()`` refuses them (``Pool is closed``), nothing drains them,
and the transports / reader tasks leak until GC.

The fix re-checks ``self._closed`` at the top of each put-loop
iteration. If ``close()`` lands during the gather, the tail routes
through the existing ``unqueued_survivors`` cleanup path and their
transports are closed deterministically. ``self._initialized`` is
only set to True when the put-loop completed without seeing
``_closed``, so a subsequent ``initialize()`` on a re-opened pool
(a future extension) does not short-circuit.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_close_race_routes_survivors_through_cleanup() -> None:
    """close() landing while gather is suspended must close every survivor.

    Models the production race: ``initialize``'s gather is suspended
    on TCP handshakes when ``close()`` runs on another task. With the
    fix, the put-loop's ``if self._closed: break`` diverts every
    connection into ``unqueued_survivors`` for deterministic cleanup.
    """
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)
    gather_started = asyncio.Event()
    close_done = asyncio.Event()

    async def _create() -> object:
        gather_started.set()
        # Suspend until close() has run, then resolve.
        await close_done.wait()
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    init_task = asyncio.create_task(pool.initialize())
    # Wait for gather to have started (and be suspended) on every node.
    await gather_started.wait()
    # Extra yields to make sure all three gather branches are parked.
    for _ in range(5):
        await asyncio.sleep(0)

    # Race close past the mid-gather initialize.
    await pool.close()
    # Now let gather resolve.
    close_done.set()

    await asyncio.wait_for(init_task, timeout=1.0)

    # INVARIANT: no survivors left in the queue of a closed pool.
    assert pool._pool.qsize() == 0, (
        f"leaked {pool._pool.qsize()} connections in closed pool's queue"
    )
    # INVARIANT: reservation accounting resolved exactly. A merely
    # non-negative _size would mask a bookkeeping slip where some
    # survivors closed but the counter stayed high.
    assert pool._size == 0, f"_size did not return to zero: {pool._size}"
    # INVARIANT: every created conn had close() called on it so the
    # transport does not orphan.
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"connection {i} was never closed (leaked transport)"


@pytest.mark.asyncio
async def test_initialize_close_race_guards_initialized_flag() -> None:
    """If close() interrupted the put-loop, _initialized stays False."""
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)
    gather_started = asyncio.Event()
    close_done = asyncio.Event()

    async def _create() -> object:
        gather_started.set()
        await close_done.wait()
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    init_task = asyncio.create_task(pool.initialize())
    await gather_started.wait()
    for _ in range(5):
        await asyncio.sleep(0)

    await pool.close()
    close_done.set()

    await asyncio.wait_for(init_task, timeout=1.0)

    # _initialized must not be True: if pool is reopened (future
    # extension) a second initialize() would skip silently otherwise.
    assert pool._initialized is False, (
        "_initialized became True despite close() interrupting the put-loop"
    )


@pytest.mark.asyncio
async def test_initialize_close_between_gather_and_phase_c_closes_all_survivors() -> None:
    """Close-during-publish race: ``close()`` lands AFTER gather
    returns successfully but BEFORE Phase C's ``async with
    self._lock:`` acquires. Phase C re-checks ``_closed`` under
    the lock and routes every survivor through the shielded close
    helper instead of publishing them into a closed pool.

    The pre-fix shape had a per-iteration ``if self._closed: break``
    inside the await-driven put-loop; that surface is gone (Phase C
    is atomic under the lock). The new structural protection is
    Phase C's single ``if self._closed:`` branch — pin it here by
    racing ``close()`` against the lock acquire via
    ``_lock.acquire`` interception.

    The existing sibling test
    ``test_initialize_close_race_routes_survivors_through_cleanup``
    covers the close-during-gather path; this test covers the
    close-during-publish-acquire path.
    """
    pool = ConnectionPool(["n1:9001", "n2:9001", "n3:9001"], min_size=3, max_size=3, timeout=0.5)

    mocks: list[MagicMock] = []
    for _ in range(3):
        m = MagicMock()
        m.close = AsyncMock()
        mocks.append(m)

    create_iter = iter(mocks)

    async def _create() -> object:
        return next(create_iter)

    pool._create_connection = _create  # type: ignore[assignment]

    # Intercept ``_lock.acquire`` so the FIRST acquire after gather
    # (Phase C's) runs ``close()`` BEFORE returning. ``close()``
    # itself uses the lock, so we drop the intercept on the first
    # call to avoid re-entry. Phase A's acquire already ran before
    # gather; only Phase C's hits the intercept.
    original_acquire = pool._lock.acquire
    intercept_fired = asyncio.Event()
    seen_acquires = 0

    async def _intercepted_acquire() -> bool:
        nonlocal seen_acquires
        seen_acquires += 1
        # Phase A is acquire #1; Phase C is acquire #2. The
        # failure-finally arm may have run another lock window
        # depending on path, but for the all-success path we expect
        # acquire #2 to be Phase C's.
        if seen_acquires == 2 and not intercept_fired.is_set():
            intercept_fired.set()
            # Restore the original acquire BEFORE running close() so
            # close()'s own lock acquire goes through normally.
            pool._lock.acquire = original_acquire
            await pool.close()
        return await original_acquire()

    pool._lock.acquire = _intercepted_acquire  # type: ignore[assignment]

    await asyncio.wait_for(pool.initialize(), timeout=1.0)
    assert intercept_fired.is_set(), "Phase C lock acquire was not intercepted"

    # _initialized stays False — Phase C saw _closed=True under the
    # lock and took the route-through-cleanup branch.
    assert pool._initialized is False
    # Every survivor closed: either via Phase C's close-routing or
    # via close()'s own drain.
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"survivor {i} was never closed"
    assert pool._closed is True
    # Reservation accounting resolved exactly.
    assert pool._size == 0, f"_size did not return to zero: {pool._size}"
