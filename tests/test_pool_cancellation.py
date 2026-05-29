"""ConnectionPool lifecycle invariants under cancellation and partial failure:
cancelled/failed acquirers and init survivors must release _size and effectively
close their connections; close() must wake parked acquirers via the close signal
(message "Pool is closed"), not the per-poll timeout."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    """Minimal DqliteConnection stand-in replicating the real close() guard:
    close() is a no-op once _pool_released/_protocol is None, so callers MUST
    close before flipping _pool_released. close_effective tracks whether close()
    actually ran (guard did not short-circuit), as opposed to merely being called."""

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
        self.close_effective = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        # Match real DqliteConnection.close guard: if the pool flipped the flag
        # first, this is a no-op and the transport leaks.
        if self._pool_released or self._protocol is None:
            return
        self.close_effective = True
        self._protocol = None  # type: ignore[assignment]

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _make_pool_with_fake_cluster(
    *,
    min_size: int = 0,
    max_size: int = 2,
    connect_impl: Any = None,
) -> tuple[ConnectionPool, list[_FakeConn]]:
    """Build a pool whose cluster.connect is stubbed; returns the pool and the
    list where created conns are appended."""
    created: list[_FakeConn] = []

    async def _default_connect(**kwargs: Any) -> _FakeConn:
        c = _FakeConn(name=f"c{len(created)}")
        created.append(c)
        return c

    impl = connect_impl if connect_impl is not None else _default_connect
    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = impl
    pool = ConnectionPool(
        addresses=["localhost:9001"],
        min_size=min_size,
        max_size=max_size,
        timeout=1.0,
        cluster=cluster,
    )
    return pool, created


class TestInitializePartialFailureClosesSurvivors:
    """If one _create_connection fails in initialize's gather, the already-
    succeeded conns must be closed (gather cancels siblings but never awaits
    their .close())."""

    async def test_survivors_closed_on_partial_init_failure(self) -> None:
        created: list[_FakeConn] = []
        call_count = [0]

        async def _flaky_connect(**kwargs: Any) -> _FakeConn:
            n = call_count[0]
            call_count[0] += 1
            if n == 2:
                raise DqliteConnectionError("simulated failure on 3rd connect")
            c = _FakeConn(name=f"c{n}")
            created.append(c)
            # First two finish quickly; the failing one loses the race.
            await asyncio.sleep(0.01)
            return c

        cluster = MagicMock(spec=ClusterClient)
        cluster.connect = _flaky_connect
        pool = ConnectionPool(
            addresses=["localhost:9001"],
            min_size=3,
            max_size=3,
            timeout=5.0,
            cluster=cluster,
        )

        with pytest.raises(DqliteConnectionError):
            await pool.initialize()

        assert pool._size == 0, (
            f"After partial-failure initialize, _size must be 0, got {pool._size}"
        )
        # Created conns must be effectively closed, not just method-called;
        # initialize() does not flip _pool_released.
        unclosed = [c for c in created if not c.close_effective]
        assert not unclosed, (
            f"Survivors of partial-failure gather must be effectively "
            f"closed; leaked: {[c.name for c in unclosed]}"
        )


class TestInitializeCancelDuringGatherDoesNotLeakCompletedConns:
    """An outer cancel while initialize's gather still has children pending
    must close the already-completed children. The unqueued_survivors list is
    populated AFTER gather returns, so a CancelledError out of gather skips the
    assignment and the finally iterates an empty list, leaking every transport."""

    async def test_initialize_cancel_during_gather_closes_completed_conns(self) -> None:
        completed: list[_FakeConn] = []
        call_idx = [0]
        completed_event = asyncio.Event()

        async def _half_fast_half_slow(**kwargs: Any) -> _FakeConn:
            i = call_idx[0]
            call_idx[0] += 1
            if i < 5:
                # Fast path: completes before the cancel fires.
                c = _FakeConn(name=f"c{i}")
                completed.append(c)
                if len(completed) == 5:
                    completed_event.set()
                return c
            # Slow path: hangs forever (cancelled).
            await asyncio.sleep(60)
            raise RuntimeError("unreachable")

        cluster = MagicMock(spec=ClusterClient)
        cluster.connect = _half_fast_half_slow
        pool = ConnectionPool(
            addresses=["localhost:9001"],
            min_size=10,
            max_size=10,
            timeout=5.0,
            cluster=cluster,
        )

        async def _wrapped() -> None:
            await pool.initialize()

        # Cancel while the slow children are still pending; wait for the fast 5
        # first so the leak is deterministic.
        init_task = asyncio.create_task(_wrapped())
        await completed_event.wait()
        await asyncio.sleep(0.01)  # let gather observe the 5 results
        init_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await init_task

        unclosed = [c for c in completed if not c.close_effective]
        assert not unclosed, (
            f"Cancel-during-gather leaked {len(unclosed)} of {len(completed)} "
            f"completed conns: {[c.name for c in unclosed]}. The pool's "
            "finally must walk gather's child tasks (not just the post-gather "
            "successes list) to reach completed-but-unqueued conns."
        )
        # _size must drain, else the pool drifts toward max_size on repeated
        # cancel-during-gather and eventually deadlocks.
        assert pool._size == 0


class TestAcquireCancellationRestoresSize:
    """A caller cancelled while parked in acquire() must not leak the
    reservation: a conn handed to the cancelled get_task must be returned to the
    pool or closed with _size decremented. Likewise for body-raised cancellation."""

    async def test_cancelled_waiter_does_not_leak_size(self) -> None:
        pool, created = _make_pool_with_fake_cluster(max_size=1)
        await pool.initialize()

        async def holder() -> None:
            async with pool.acquire():
                await asyncio.Event().wait()  # never release

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        async def waiter() -> None:
            async with pool.acquire():
                pass

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)

        # Cancel the waiter while it is parked on the queue.
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        assert pool._size == 1, (
            f"After cancelling a parked acquirer, _size must still be 1 "
            f"(the holder), got {pool._size}"
        )

        holder_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder_task
        await pool.close()

    async def test_reset_connection_failure_releases_reservation(self) -> None:
        """When _reset_connection fails (e.g. ROLLBACK raises), the reservation
        must be released and the connection closed, never leaked."""

        class _RollbackFailingConn(_FakeConn):
            async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
                if sql.upper().strip().startswith("ROLLBACK"):
                    raise DqliteConnectionError("ROLLBACK failed: connection lost")
                return (0, 0)

        conn = _RollbackFailingConn(name="rollback-fails")
        created_holder: list[_FakeConn] = [conn]

        async def _connect_once(**kwargs: Any) -> _FakeConn:
            return created_holder[0]

        cluster = MagicMock(spec=ClusterClient)
        cluster.connect = _connect_once
        pool = ConnectionPool(
            addresses=["localhost:9001"],
            min_size=0,
            max_size=1,
            timeout=5.0,
            cluster=cluster,
        )

        class _BodyError(Exception):
            pass

        with pytest.raises(_BodyError):
            async with pool.acquire() as c:
                # Simulate BEGIN so the pool thinks there is a tx to roll back.
                c._in_transaction = True
                raise _BodyError()

        # Reservation must be back and the conn effectively closed: the pool must
        # close() BEFORE setting _pool_released or close() no-ops and leaks.
        assert pool._size == 0, (
            f"After _reset_connection failure, _size must be 0 (reservation "
            f"released), got {pool._size}"
        )
        assert conn.close_called, "close() must be invoked"
        assert conn.close_effective, (
            "close() must actually run — not be short-circuited by the "
            "_pool_released guard. The pool must close() BEFORE setting "
            "_pool_released=True, or the transport leaks (critical bug "
            "flagged by concurrency review)."
        )

        await pool.close()


class TestCloseWakesAllWaiters:
    """pool.close() must wake every parked acquire() with DqliteConnectionError
    promptly; the clear()-then-wait pattern can erase close()'s set() and stall
    waiters until timeout."""

    async def test_close_wakes_parked_waiters(self) -> None:
        pool, _ = _make_pool_with_fake_cluster(max_size=1)
        await pool.initialize()

        async def holder() -> None:
            async with pool.acquire():
                await asyncio.Event().wait()

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        async def waiter() -> Any:
            try:
                async with pool.acquire():
                    return "got-conn"
            except DqliteConnectionError as e:
                return e

        N = 5
        waiters = [asyncio.create_task(waiter()) for _ in range(N)]
        await asyncio.sleep(0.05)

        # The message is the invariant: a waiter that hit the closed_event.clear()
        # race wakes via the per-poll timeout and raises "Timed out..." instead.
        # Assert on message text, not elapsed time (CI latency is variable).
        await pool.close()

        results = await asyncio.wait_for(
            asyncio.gather(*waiters, return_exceptions=True),
            timeout=2.0,
        )

        for r in results:
            assert isinstance(r, DqliteConnectionError), (
                f"Parked acquirer should see DqliteConnectionError after close(), got {r!r}"
            )
            assert "closed" in str(r).lower(), (
                f"Waiters must wake via close signal (message: 'Pool is closed'), "
                f"not via the per-poll timeout path — the closed_event.clear() "
                f"race would cause the latter. Got: {r!s}"
            )

        holder_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder_task


class TestResetConnectionNarrowsException:
    """_reset_connection's ROLLBACK catch must only treat transport-level errors
    as "drop the connection"; programming errors and cancellation must propagate."""

    async def test_rollback_programming_error_propagates(self) -> None:
        class _BoomConn(_FakeConn):
            async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
                if sql.upper().strip().startswith("ROLLBACK"):
                    raise AttributeError("stale attribute — programmer bug")
                return (0, 0)

        conn = _BoomConn(name="boom-rollback")

        async def _connect_once(**kwargs: Any) -> _FakeConn:
            return conn

        cluster = MagicMock(spec=ClusterClient)
        cluster.connect = _connect_once
        pool = ConnectionPool(
            addresses=["localhost:9001"],
            min_size=0,
            max_size=1,
            timeout=5.0,
            cluster=cluster,
        )

        class _BodyDone(Exception):
            pass

        with pytest.raises(AttributeError, match="stale attribute"):
            async with pool.acquire() as c:
                c._in_transaction = True
                raise _BodyDone()

        await pool.close()


class TestInitializeCleanupNarrowsException:
    """initialize() partial-failure cleanup must not swallow programming errors
    during conn.close(); a bare suppress(BaseException) masks bugs and cancels."""

    async def test_cleanup_programming_error_propagates(self) -> None:
        call_count = [0]

        class _CloseBoomConn(_FakeConn):
            async def close(self) -> None:
                raise AttributeError("close bug in survivor")

        async def _flaky_connect(**kwargs: Any) -> _FakeConn:
            n = call_count[0]
            call_count[0] += 1
            if n == 1:
                # Second connect fails so the cleanup path runs on the first.
                await asyncio.sleep(0.01)
                raise DqliteConnectionError("second connect failed")
            c = _CloseBoomConn(name=f"c{n}")
            return c

        cluster = MagicMock(spec=ClusterClient)
        cluster.connect = _flaky_connect
        pool = ConnectionPool(
            addresses=["localhost:9001"],
            min_size=2,
            max_size=2,
            timeout=5.0,
            cluster=cluster,
        )

        with pytest.raises(AttributeError, match="close bug"):
            await pool.initialize()
