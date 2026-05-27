"""Pins for the post-fix 3-phase ``ConnectionPool.initialize()``
state machine:

* Phase A reserves under ``_lock`` (no awaits) and either flips
  ``_initialized`` immediately when ``min_size <= 0`` or sets
  ``_initializing`` + creates ``_initialize_done_event`` for the
  first caller.
* Phase B runs ``await asyncio.gather(*create_tasks)`` LOCK-FREE
  so sibling tasks can make progress against the partially-warmed
  pool. The Phase B failure-finally clears ``_initializing`` and
  signals the event so secondary callers can wake and retry.
* Phase C re-acquires ``_lock`` (no awaits) for the atomic
  publish via ``put_nowait``; re-checks ``_closed`` and routes
  survivors through the shielded close helper if a sibling
  ``close()`` flipped the flag during Phase B.

Tests in this file cover the discriminating behaviours the
3-phase shape introduces: secondary-caller event coordination,
fork-safety nulling, failure-retry contract, and Phase B cancel
unwind. The sibling Test A pin
(``test_pool_initialize_does_not_block_concurrent_acquire.py``)
covers the lock-free Phase B invariant; the
``test_pool_initialize_close_race.py`` suite covers the
close-race invariants.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self) -> None:
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol: MagicMock | None = MagicMock()
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self._protocol = None

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _pool_with_factory(factory: Any, *, min_size: int, max_size: int) -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = factory
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=min_size,
        max_size=max_size,
        timeout=10.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_concurrent_initialize_creates_exactly_min_size_connections() -> None:
    """Two concurrent ``initialize()`` calls must produce exactly
    ``min_size`` connections (not 2× ``min_size``). The second
    caller observes ``_initializing=True`` under the lock, parks on
    ``_initialize_done_event`` outside the lock, and on wake the
    recursive ``return await self.initialize()`` finds
    ``_initialized=True`` and short-circuits.
    """
    create_count = 0

    async def _factory(**kwargs: Any) -> _FakeConn:
        nonlocal create_count
        create_count += 1
        await asyncio.sleep(0.05)
        return _FakeConn()

    pool = _pool_with_factory(_factory, min_size=4, max_size=10)
    try:
        await asyncio.gather(pool.initialize(), pool.initialize())
        assert create_count == 4, (
            f"expected exactly 4 connections, got {create_count}; "
            "second concurrent initialize() should have parked on the "
            "event instead of re-running the warm-up gather"
        )
        assert pool._initialized is True
        assert pool._pool.qsize() == 4
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_secondary_caller_wakes_promptly_when_first_completes() -> None:
    """A secondary caller scheduled WHILE the first is in Phase B
    must wake within ~the first's wall-clock + a small delta, NOT
    block until its own timeout.
    """
    first_done = asyncio.Event()

    async def _factory(**kwargs: Any) -> _FakeConn:
        await asyncio.sleep(0.2)
        first_done.set()
        return _FakeConn()

    pool = _pool_with_factory(_factory, min_size=2, max_size=4)
    try:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.gather(pool.initialize(), pool.initialize())
        elapsed = loop.time() - start
        # First caller's gather takes ~0.2s; second caller wakes on
        # event-set in the same Phase C. Total wall-clock should be
        # ~0.2s + scheduling slack, definitely under 0.5s.
        assert elapsed < 0.5, (
            f"concurrent initialize took {elapsed:.3f}s; secondary "
            f"caller may have re-run its own warm-up gather instead of "
            f"parking on the event"
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_failed_initialize_leaves_pool_retryable() -> None:
    """A Phase B failure must clear ``_initializing``, signal the
    event, and leave ``_initialized=False``. A subsequent
    ``initialize()`` call re-enters Phase A cleanly and runs a
    fresh warm-up gather.
    """
    attempt = 0

    async def _factory(**kwargs: Any) -> _FakeConn:
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            raise DqliteConnectionError("synthetic warm-up failure")
        return _FakeConn()

    pool = _pool_with_factory(_factory, min_size=2, max_size=4)
    try:
        with pytest.raises((DqliteConnectionError, BaseExceptionGroup)):
            await pool.initialize()
        assert pool._initialized is False
        assert pool._initializing is False
        assert pool._initialize_done_event is None
        assert pool._size == 0

        # Retry: third+fourth factory calls succeed.
        await pool.initialize()
        assert pool._initialized is True
        assert pool._pool.qsize() == 2
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_zero_min_size_short_circuits_under_lock_without_event() -> None:
    """``min_size=0`` is a no-warm-up pool. Phase A flips
    ``_initialized`` under the lock and returns without creating
    the event — there is no Phase B to run.
    """

    async def _factory(**kwargs: Any) -> _FakeConn:
        raise AssertionError("factory should not be called for min_size=0")

    pool = _pool_with_factory(_factory, min_size=0, max_size=4)
    try:
        await pool.initialize()
        assert pool._initialized is True
        assert pool._initializing is False
        # Event is never created on the no-warm-up path because no
        # secondary caller can observe ``_initializing``.
        assert pool._initialize_done_event is None
        assert pool._size == 0
    finally:
        await pool.close()


def test_fork_shortcut_nulls_initialize_done_event_and_flag() -> None:
    """A child-side ``close()`` (post-fork, pid mismatch) must null
    both ``_initialize_done_event`` (loop-bound to the parent) and
    ``_initializing`` so a subsequent child-side ``initialize()``
    does not park on a parent-loop event forever.

    Mirrors the existing fork-shortcut discipline for ``_close_done``
    / ``_closed_event``.
    """

    async def _factory(**kwargs: Any) -> _FakeConn:
        return _FakeConn()

    pool = _pool_with_factory(_factory, min_size=2, max_size=4)
    # Forge mid-initialize state without actually running the loop.
    fake_loop = asyncio.new_event_loop()
    try:
        pool._initialize_done_event = asyncio.Event()
        pool._initializing = True
        # Force the fork-pid mismatch.
        pool._creator_pid = os.getpid() + 999_999

        # Drive the fork-shortcut branch by calling close() in a
        # fresh loop (so the get_current_pid mismatch fires before
        # any loop-bound primitive is touched).
        async def _drive() -> None:
            await pool.close()

        fake_loop.run_until_complete(_drive())

        assert pool._initialize_done_event is None, (
            "fork shortcut must null _initialize_done_event so a child-"
            "side initialize() does not park on a parent-loop event"
        )
        assert pool._initializing is False, (
            "fork shortcut must clear _initializing so a child-side "
            "initialize() takes the first-caller path instead of "
            "parking on the (now-nulled) event"
        )
    finally:
        fake_loop.close()
