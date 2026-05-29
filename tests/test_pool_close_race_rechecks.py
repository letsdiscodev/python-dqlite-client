"""Pin the in-lock ``_closed`` rechecks in ``acquire()`` (reservation and
at-capacity-wait blocks) plus the ``async with pool:`` lifecycle.

The rechecks fire when ``close()`` lands after the outer ``_closed`` check but
before the lock is acquired, simulated via a Lock that flips ``_closed``.
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
        self._protocol = None  # type: ignore[assignment]

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
    """Lock whose ``acquire()`` flips ``target._closed`` True on the
    ``flip_on``-th call (lets us flip selectively per acquire site)."""

    class _CountedFlipLock(asyncio.Lock):
        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        async def acquire(self) -> bool:  # type: ignore[override]
            result = await super().acquire()
            self._n += 1
            if self._n == flip_on:
                target._closed = True
            return result

    return _CountedFlipLock()


class TestAcquireReservationLockClosedRecheck:
    async def test_close_lands_during_reservation_lock_acquire(self) -> None:
        """If ``close()`` flips ``_closed`` while acquiring the reservation lock,
        the in-lock recheck must raise rather than bump ``_size``."""

        async def _factory(**kwargs: Any) -> _FakeConn:  # pragma: no cover - never reached
            return _FakeConn()

        pool = _pool_with_factory(_factory, max_size=2)
        await pool.initialize()  # min_size=0: empty queue, enter reservation block
        pool._lock = _counted_flipping_lock(pool, flip_on=1)

        with pytest.raises(DqliteConnectionError, match="Pool is closed"):
            async with pool.acquire():
                pass  # pragma: no cover - acquire must raise


class TestAcquireAtCapacityWaitLockClosedRecheck:
    async def test_close_lands_during_wait_state_lock_acquire(self) -> None:
        """At capacity, if ``close()`` flips ``_closed`` while acquiring the
        wait-block lock, the recheck must raise rather than park on a queue
        that will never fill."""
        c = _FakeConn(name="c")
        connections = iter([c])

        async def _factory(**kwargs: Any) -> _FakeConn:
            return next(connections)

        pool = _pool_with_factory(_factory, max_size=1)
        await pool.initialize()

        # Pre-acquire the only slot so the next acquire enters the wait branch.
        cm = pool.acquire()
        await cm.__aenter__()

        # n=1 is the reservation lock (must not flip — falls through to the
        # wait branch); n=2 is the wait-branch lock — flip there.
        pool._lock = _counted_flipping_lock(pool, flip_on=2)

        with pytest.raises(DqliteConnectionError, match="Pool is closed"):
            async with pool.acquire():
                pass  # pragma: no cover - acquire must raise

        await cm.__aexit__(None, None, None)  # release held conn so close() can drain


class TestPoolAsyncWithLifecycle:
    async def test_async_with_initializes_and_closes_with_mock_cluster(self) -> None:
        """``async with pool:`` runs ``__aenter__`` (initialize) and
        ``__aexit__`` (close)."""
        c = _FakeConn()

        async def _factory(**kwargs: Any) -> _FakeConn:
            return c

        pool = _pool_with_factory(_factory, max_size=1)

        async with pool as entered:
            assert entered is pool
            assert pool._initialized is True
            assert pool._closed is False

        assert pool._closed is True


class TestCreatePoolBody:
    async def test_create_pool_returns_initialized_pool_with_mock_cluster(self) -> None:
        """``create_pool(...)`` awaits ``initialize()`` before returning."""
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
