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

    pool._create_connection = _create  # type: ignore[method-assign]

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

    pool._create_connection = _create  # type: ignore[method-assign]

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
