"""Pin close-race rechecks in ``ConnectionPool.acquire()`` and the
``async with pool:`` lifecycle reported as uncovered by
``pytest --cov``.

Lines covered:

- 495 — outer lock-recheck: ``async with self._lock`` for the
  reservation block, then ``if self._closed: raise``.
- 523 — at-capacity wait recheck: ``async with self._lock`` for the
  state-change wait block, then ``if self._closed: raise``.
- 918-919, 927 — ``__aenter__`` / ``__aexit__`` (the async-with form
  of pool lifecycle, exercised here with a mocked cluster so no
  cluster fixture dependency).

The L495 / L523 paths fire when ``close()`` lands AFTER the outer
``_closed`` check but BEFORE the lock acquisition completes. Driven
deterministically via an ``asyncio.Lock`` subclass that flips
``_closed`` mid-``acquire()`` — a controlled simulation of the race
window the rechecks defend against.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
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

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self._protocol = None

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _pool_with_factory(factory: Any, *, max_size: int = 1) -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = factory
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=max_size,
        timeout=0.5,
        cluster=cluster,
    )


def _counted_flipping_lock(target: ConnectionPool, flip_on: int) -> asyncio.Lock:
    """Return an ``asyncio.Lock`` whose ``acquire()`` flips
    ``target._closed`` to True on the ``flip_on``-th acquire.

    A counter lets us flip selectively (e.g. only on the second
    acquire so the first reservation succeeds before the close-race
    fires)."""

    class _CountedFlipLock(asyncio.Lock):
        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        async def acquire(self) -> bool:
            result = await super().acquire()
            self._n += 1
            if self._n == flip_on:
                target._closed = True
            return result

    return _CountedFlipLock()


# ---------------------------------------------------------------------------
# pool.py:495 — reservation lock-recheck
# ---------------------------------------------------------------------------


class TestAcquireReservationLockClosedRecheck:
    async def test_close_lands_during_reservation_lock_acquire(self) -> None:
        """``acquire()``'s reservation block enters ``self._lock``;
        if ``close()`` flips ``_closed`` between the outer check and
        the lock acquisition, the in-lock recheck must surface it
        as ``DqliteConnectionError("Pool is closed")`` rather than
        proceeding to bump ``_size``. Drives pool.py:495."""

        async def _factory(**kwargs: Any) -> _FakeConn:  # pragma: no cover - never reached
            return _FakeConn()

        pool = _pool_with_factory(_factory, max_size=2)
        # Initialize with min_size=0 so queue is empty and we enter
        # the reservation block.
        await pool.initialize()
        # Replace the lock with a flipping variant that flips on
        # the first acquire (the reservation lock entry).
        pool._lock = _counted_flipping_lock(pool, flip_on=1)

        with pytest.raises(DqliteConnectionError, match="Pool is closed"):
            async with pool.acquire():
                pass  # pragma: no cover - acquire must raise


# ---------------------------------------------------------------------------
# pool.py:523 — at-capacity wait lock-recheck
# ---------------------------------------------------------------------------


class TestAcquireAtCapacityWaitLockClosedRecheck:
    async def test_close_lands_during_wait_state_lock_acquire(self) -> None:
        """When the pool is at capacity, ``acquire()`` enters
        ``self._lock`` to clear the closed-event before parking on
        ``self._pool.get()``. If ``close()`` flips ``_closed`` between
        the outer check and this lock acquisition, the recheck must
        surface as ``DqliteConnectionError`` rather than parking on
        a queue that will never fill. Drives pool.py:523."""
        c = _FakeConn(name="c")
        connections = iter([c])

        async def _factory(**kwargs: Any) -> _FakeConn:
            return next(connections)

        pool = _pool_with_factory(_factory, max_size=1)
        await pool.initialize()

        # Pre-acquire the only slot so the next acquire enters the
        # at-capacity wait branch.
        cm = pool.acquire()
        await cm.__aenter__()

        # Replace the lock with a counted flipping variant. The
        # second acquire's first lock entry is the reservation lock
        # (n=1 — must NOT flip; flowing through reservation to
        # ``reserved=False`` and dropping out to the wait branch).
        # The wait branch's lock acquire is n=2 — flip there.
        pool._lock = _counted_flipping_lock(pool, flip_on=2)

        with pytest.raises(DqliteConnectionError, match="Pool is closed"):
            async with pool.acquire():
                pass  # pragma: no cover - acquire must raise

        # Cleanup: release the held conn so close() can drain.
        await cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# pool.py:918-919, 927 — async-with lifecycle (mocked cluster — no
# cluster fixture dependency)
# ---------------------------------------------------------------------------


class TestPoolAsyncWithLifecycle:
    async def test_async_with_initializes_and_closes_with_mock_cluster(self) -> None:
        """``async with ConnectionPool(...) as pool:`` drives
        ``__aenter__`` (which calls ``initialize()``) and
        ``__aexit__`` (which calls ``close()``). Drives pool.py
        lines 918-919 and 927.

        Uses ``min_size=0`` + a mocked cluster to avoid the
        cluster-fixture dependency that gates the analogous
        ``create_pool`` integration test."""
        c = _FakeConn()

        async def _factory(**kwargs: Any) -> _FakeConn:
            return c

        pool = _pool_with_factory(_factory, max_size=1)

        async with pool as entered:
            assert entered is pool
            assert pool._initialized is True
            assert pool._closed is False

        # __aexit__ ran → close() ran → pool is closed.
        assert pool._closed is True


# ---------------------------------------------------------------------------
# __init__.py:143-157 — create_pool body (mocked cluster)
# ---------------------------------------------------------------------------


class TestCreatePoolBody:
    async def test_create_pool_returns_initialized_pool_with_mock_cluster(self) -> None:
        """``create_pool(...)`` constructs a ``ConnectionPool`` and
        awaits ``initialize()`` before returning. The integration
        version of this test (in ``tests/integration/test_query_raw_apis.py``)
        is skipped pending cluster-fixture work — exercise the body
        here against a mocked cluster so the public ``create_pool``
        constructor is covered."""
        from dqliteclient import create_pool

        c = _FakeConn()

        async def _factory(**kwargs: Any) -> _FakeConn:
            return c

        cluster = MagicMock(spec=ClusterClient)
        cluster.connect = _factory

        pool = await create_pool(min_size=0, max_size=1, cluster=cluster)
        try:
            assert pool._initialized is True
            assert pool._closed is False
        finally:
            await pool.close()
