"""Pin defensive cleanup paths in ``ConnectionPool`` reported as
uncovered by ``pytest --cov``.

Lines covered (pre-pragma):

- 297     — ``initialize`` partial-cleanup close error swallow when
  a survivor's ``close()`` raises ``_POOL_CLEANUP_EXCEPTIONS``.
- 354-355 — ``initialize`` unqueued-survivor close error swallow.
- 635-637 — ``acquire`` reservation release on ``_create_connection``
  failure during the drain-then-create stale-conn path.
- 675-678 — ``acquire`` release path's ``QueueFull`` → close branch
  in the normal (non-cancelled) return-to-pool flow.

Each line is a transport-cleanup defensive path. A regression that
swallowed these silently would leak connections on the failure
shapes they guard.
"""

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
    """Minimal stand-in tracking close() and is_connected. Mirrors
    the shape used by ``test_pool_acquire_queuefull.py``."""

    def __init__(self, name: str = "fake", *, alive: bool = True) -> None:
        self.name = name
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol = MagicMock() if alive else None
        if alive:
            self._protocol._writer = MagicMock()
            self._protocol._writer.transport = MagicMock()
            self._protocol._writer.transport.is_closing = lambda: False
            self._protocol._reader = MagicMock()
            self._protocol._reader.at_eof = lambda: False
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


# ---------------------------------------------------------------------------
# pool.py:297 — initialize() partial-cleanup close error swallow
# ---------------------------------------------------------------------------


class TestInitializePartialCleanupSwallowsCloseError:
    async def test_close_error_on_survivor_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Gather has 1 success + 1 failure; the survivor's ``close()``
        raises ``OSError`` (a member of ``_POOL_CLEANUP_EXCEPTIONS``).
        The pool must DEBUG-log "partial-cleanup close error" and
        still surface the original failure to the caller."""
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
        assert any("partial-cleanup close error" in rec.message for rec in caplog.records), (
            f"expected DEBUG record; got {[r.message for r in caplog.records]!r}"
        )


# ---------------------------------------------------------------------------
# pool.py:354-355 — initialize() unqueued-survivor close error swallow
# ---------------------------------------------------------------------------


class TestInitializeUnqueuedSurvivorCloseErrorSwallow:
    async def test_close_error_in_unqueued_finally_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Drive the ``finally``-block's survivor-close path: gather
        all-success, but the put-loop is interrupted (``put`` raises
        ``CancelledError`` after the first put), and the unqueued
        tail's ``close()`` raises an ``_POOL_CLEANUP_EXCEPTIONS``
        member. The pool must DEBUG-log
        "unqueued-survivor close error"."""
        a = _FakeConn(name="a")
        b = _FakeConn(name="b")
        b.close_raises = OSError("transport gone")

        connections = iter([a, b])

        async def _factory(**kwargs: Any) -> _FakeConn:
            return next(connections)

        pool = _pool_with_factory(_factory, min_size=2, max_size=2)

        original_put = pool._pool.put
        puts = 0

        async def _intercept_put(item: object) -> None:
            nonlocal puts
            await original_put(item)
            puts += 1
            if puts == 1:
                # Cancel the in-flight initialize() task at the next
                # await point — the put-loop's next iteration will
                # observe the cancellation and the finally enters
                # the unqueued-survivor cleanup with ``b`` still in
                # the unqueued list.
                raise asyncio.CancelledError

        pool._pool.put = _intercept_put  # type: ignore[method-assign]

        with (
            caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"),
            pytest.raises(asyncio.CancelledError),
        ):
            await pool.initialize()

        assert b.close_called >= 1, "unqueued survivor close was never attempted"
        assert any("unqueued-survivor close error" in rec.message for rec in caplog.records), (
            f"expected DEBUG record; got {[r.message for r in caplog.records]!r}"
        )


# ---------------------------------------------------------------------------
# pool.py:635-637 — drain-then-create reservation release on failure
# ---------------------------------------------------------------------------


class TestAcquireReservationReleasedOnDrainThenCreateFailure:
    async def test_create_failure_after_drain_releases_reservation(self) -> None:
        """When ``acquire()`` finds a stale conn, drains the queue,
        and tries to create a fresh one, a failure during creation
        must release the reservation. Without this, ``_size`` stays
        inflated and the pool wedges at capacity."""
        # Pre-populate pool with one stale connection.
        stale = _FakeConn(name="stale", alive=False)
        attempts: list[str] = []

        async def _factory(**kwargs: Any) -> _FakeConn:
            attempts.append("create")
            raise DqliteConnectionError("create failed after drain")

        pool = _pool_with_factory(_factory, min_size=0, max_size=1)
        # Manually inject the stale conn + reservation as if a previous
        # acquire+release cycle had populated it.
        await pool._pool.put(stale)
        pool._size = 1

        size_before = pool._size
        with pytest.raises(DqliteConnectionError, match="create failed after drain"):
            async with pool.acquire():
                pass  # pragma: no cover - acquire raises before yielding

        # Reservation must have been released. Without the
        # ``_release_reservation()`` at L636, ``_size`` stays at 1
        # and the pool is permanently at capacity.
        assert pool._size == size_before - 1, (
            f"reservation not released after drain-then-create failure: "
            f"size before={size_before}, after={pool._size}"
        )


# ---------------------------------------------------------------------------
# pool.py:675-678 — release-path QueueFull closes the conn
# ---------------------------------------------------------------------------


class _UserError(Exception):
    """User-code exception used to drive the ``except BaseException:``
    cleanup arm of ``acquire()`` deterministically."""


class TestReleaseQueueFullClosesConnection:
    async def test_release_queuefull_closes_conn_in_baseexception_arm(self) -> None:
        """In the ``except BaseException:`` arm of ``acquire()``
        (user code raised mid-context-body), a healthy conn whose
        ``put_nowait`` raises ``QueueFull`` (invariant violation)
        must be closed and have ``_pool_released`` set rather than
        silently dropped. Drives pool.py:675-678."""
        c = _FakeConn(name="c")
        connections = iter([c])

        async def _factory(**kwargs: Any) -> _FakeConn:
            return next(connections)

        pool = _pool_with_factory(_factory, min_size=0, max_size=1)

        # Acquire, then have the body raise — the cleanup arm
        # routes through the QueueFull branch when put_nowait fails.
        with pytest.raises(_UserError):
            async with pool.acquire() as got:
                assert got is c
                pool._pool.put_nowait = MagicMock(side_effect=asyncio.QueueFull)
                raise _UserError("user code raised")

        assert c.close_called >= 1, (
            "QueueFull on release must close the conn, not silently drop the reference"
        )
        assert c._pool_released is True
