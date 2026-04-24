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


@pytest.mark.asyncio
async def test_initialize_close_between_put_iterations_closes_tail_survivors() -> None:
    """Put-loop mid-iteration break: when ``close()`` lands AFTER at
    least one put succeeded but BEFORE the remaining successes have
    been enqueued, the tail routes through ``unqueued_survivors``
    cleanup. Every survivor must be closed and reservation accounting
    must resolve to zero.

    The existing tests in this module cover the shape where ``close()``
    lands before ANY put — the whole batch takes the break path at
    the top of the put-loop. This test covers the intermediate shape:
    one put committed, close lands, remaining tail goes through
    unqueued cleanup.
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

    pool._create_connection = _create  # type: ignore[method-assign]

    # Intercept the queue put so that exactly the first put succeeds,
    # then close() fires before the second put iteration runs.
    original_put = pool._pool.put
    puts_done = 0
    close_fired = asyncio.Event()

    async def _intercept_put(item: object) -> None:
        nonlocal puts_done
        await original_put(item)
        puts_done += 1
        if puts_done == 1:
            # Land the close() between put-loop iterations.
            await pool.close()
            close_fired.set()

    pool._pool.put = _intercept_put  # type: ignore[method-assign]

    await asyncio.wait_for(pool.initialize(), timeout=1.0)
    assert close_fired.is_set(), "close() did not land between put-iterations"

    # _initialized must stay False — the put-loop was interrupted by
    # a concurrent close().
    assert pool._initialized is False
    # Every mock must have been closed: the first entered the queue
    # and was drained by close()'s _drain_idle; the remaining tail
    # went through the unqueued_survivors cleanup path. Both routes
    # end in ``conn.close()`` being awaited at least once.
    for i, m in enumerate(mocks):
        assert m.close.await_count >= 1, f"survivor {i} was never closed"
    # Pool must be in the closed state (no further acquires possible).
    assert pool._closed is True
