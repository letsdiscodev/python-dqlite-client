"""Cover the defensive transport-cleanup paths in ``ConnectionPool`` (initialize
partial cleanup, drain-then-create reservation release, release-path QueueFull close)
whose regressions would silently leak connections."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    """Minimal stand-in tracking close() and is_connected."""

    def __init__(self, name: str = "fake", *, alive: bool = True) -> None:
        self.name = name
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol = MagicMock() if alive else None
        if alive:
            self._protocol._writer = MagicMock()  # type: ignore[union-attr]
            self._protocol._writer.transport = MagicMock()  # type: ignore[union-attr]
            self._protocol._writer.transport.is_closing = lambda: False  # type: ignore[union-attr]
            self._protocol._reader = MagicMock()  # type: ignore[union-attr]
            self._protocol._reader.at_eof = lambda: False  # type: ignore[union-attr]
        self.close_called = 0
        self.close_raises: BaseException | None = None

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called += 1
        if self.close_raises is not None:
            raise self.close_raises
        self._protocol = None

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _pool_with_factory(
    factory: Any,
    *,
    min_size: int = 0,
    max_size: int = 2,
) -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = factory
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=min_size,
        max_size=max_size,
        timeout=1.0,
        cluster=cluster,
    )


class TestInitializePartialCleanupSwallowsCloseError:
    async def test_close_error_on_survivor_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """1 success + 1 failure; the survivor's close() raises OSError. The pool must
        DEBUG-log the close error and still surface the original failure."""
        survivor = _FakeConn(name="survivor")
        survivor.close_raises = OSError("transport already gone")
        attempts: list[int] = []

        async def _factory(**kwargs: Any) -> _FakeConn:
            attempts.append(1)
            if len(attempts) == 1:
                return survivor
            raise DqliteConnectionError("create failed")

        pool = _pool_with_factory(_factory, min_size=2, max_size=2)
        with (
            caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"),
            pytest.raises(DqliteConnectionError, match="create failed"),
        ):
            await pool.initialize()

        assert survivor.close_called >= 1, "survivor close was never attempted"
        # The success-cleanup walk routes through _initialize_close_unqueued, which
        # logs under the "unqueued-survivor close error" key.
        assert any("unqueued-survivor close error" in rec.message for rec in caplog.records), (
            f"expected DEBUG record; got {[r.message for r in caplog.records]!r}"
        )


class TestInitializeUnqueuedSurvivorCloseErrorSwallow:
    async def test_close_error_in_unqueued_finally_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase C exception path: gather all-success, then put_nowait raises. The
        unqueued tail's close() raises; the pool must DEBUG-log the close error and
        re-raise the original publish failure."""
        a = _FakeConn(name="a")
        b = _FakeConn(name="b")
        b.close_raises = OSError("transport gone")

        connections = iter([a, b])

        async def _factory(**kwargs: Any) -> _FakeConn:
            return next(connections)

        pool = _pool_with_factory(_factory, min_size=2, max_size=2)

        # Raise on the second put_nowait so the first conn is queued and the second
        # triggers the Phase C exception path.
        original_put_nowait = pool._pool.put_nowait
        puts = 0

        def _intercept_put_nowait(item: object) -> None:
            nonlocal puts
            puts += 1
            if puts == 2:
                raise asyncio.QueueFull
            original_put_nowait(item)  # type: ignore[arg-type]

        pool._pool.put_nowait = _intercept_put_nowait

        with (
            caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"),
            pytest.raises(asyncio.QueueFull),
        ):
            await pool.initialize()

        assert b.close_called >= 1, "unqueued survivor close was never attempted"
        assert any("unqueued-survivor close error" in rec.message for rec in caplog.records), (
            f"expected DEBUG record; got {[r.message for r in caplog.records]!r}"
        )


class TestAcquireReservationReleasedOnDrainThenCreateFailure:
    async def test_create_failure_after_drain_releases_reservation(self) -> None:
        """When acquire() drains a stale conn then fails to create a fresh one, the
        reservation must be released or _size stays inflated and the pool wedges."""
        stale = _FakeConn(name="stale", alive=False)
        attempts: list[str] = []

        async def _factory(**kwargs: Any) -> _FakeConn:
            attempts.append("create")
            raise DqliteConnectionError("create failed after drain")

        pool = _pool_with_factory(_factory, min_size=0, max_size=1)
        # Inject the stale conn + reservation as if a prior acquire+release populated it.
        await pool._pool.put(stale)  # type: ignore[arg-type]
        pool._size = 1

        size_before = pool._size
        with pytest.raises(DqliteConnectionError, match="create failed after drain"):
            async with pool.acquire():
                pass  # pragma: no cover - acquire raises before yielding

        assert pool._size == size_before - 1, (
            f"reservation not released after drain-then-create failure: "
            f"size before={size_before}, after={pool._size}"
        )


class _UserError(Exception):
    """Drives the ``except BaseException:`` cleanup arm of ``acquire()``."""


class TestReleaseQueueFullClosesConnection:
    async def test_release_queuefull_closes_conn_in_baseexception_arm(self) -> None:
        """In acquire()'s except-BaseException arm, a healthy conn whose put_nowait raises
        QueueFull must be closed and have _pool_released set, not silently dropped."""
        c = _FakeConn(name="c")
        connections = iter([c])

        async def _factory(**kwargs: Any) -> _FakeConn:
            return next(connections)

        pool = _pool_with_factory(_factory, min_size=0, max_size=1)

        with pytest.raises(_UserError):
            async with pool.acquire() as got:
                assert got is c
                pool._pool.put_nowait = MagicMock(side_effect=asyncio.QueueFull)
                raise _UserError("user code raised")

        assert c.close_called >= 1, (
            "QueueFull on release must close the conn, not silently drop the reference"
        )
        assert c._pool_released is True
