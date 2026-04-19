"""Pool-level cancellation and partial-failure invariants.

Pins the ``ConnectionPool`` lifecycle guarantees against cancellation
and partial failures:

- Cancelling a parked ``acquire()`` caller must not leak ``_size``.
  ``_size`` must reflect reality at all times.
- ``pool.close()`` must wake every parked acquirer via the close
  signal, not via the per-poll timeout. Waking via timeout would
  produce "Timed out waiting for a connection" instead of
  "Pool is closed" — the message is the observable signal.
- ``_reset_connection`` failure during cleanup must release the
  reservation and effectively close the connection. ``_FakeConn``
  replicates the real ``DqliteConnection.close`` early-return guard
  on ``_pool_released``, so any regression that flips the flag
  before calling close() surfaces as a failed ``close_effective``
  assertion.
- ``initialize()`` partial failure: sibling connections that already
  succeeded must be effectively closed before ``gather()`` propagates
  the first failure. The previous default
  ``return_exceptions=False`` cancelled siblings but did NOT close
  them, leaking transports.

No cluster needed — the pool is instantiated with a mock cluster whose
``connect`` coroutine is stubbed per test.
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
    """Minimal DqliteConnection stand-in the pool accepts.

    Matches the real connection's close() semantics: the early-return
    guard on ``_pool_released`` / ``_protocol is None`` means close() is
    a no-op when either is set. Pool callers MUST close before flipping
    ``_pool_released`` — the fake replicates that guard so a test-side
    mistake surfaces the same symptom as a bug in the real pool code
    (transport leaked, socket still open).

    ``close_effective`` tracks whether close() actually did work (the
    guard did NOT short-circuit). Tests assert on this rather than
    ``close_called`` when they care about "the transport actually
    closed", not "close() method was invoked".
    """

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
        self.close_effective = False  # did close() do actual work?

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        # Match real DqliteConnection.close guard. If the pool flipped
        # the flag first, this is a no-op and the transport leaks — the
        # exact bug the pool code must not have.
        if self._pool_released or self._protocol is None:
            return
        self.close_effective = True
        self._protocol = None

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _make_pool_with_fake_cluster(
    *,
    min_size: int = 0,
    max_size: int = 2,
    connect_impl: Any = None,
) -> tuple[ConnectionPool, list[_FakeConn]]:
    """Build a pool whose cluster.connect is stubbed.

    ``connect_impl`` is an async callable returning a _FakeConn (or
    raising). Default returns a fresh _FakeConn each call. Returns the
    pool and the list where created conns are appended.
    """
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
    """If a single _create_connection fails in initialize's gather,
    the connections that already succeeded must be closed. Previously
    they leaked (asyncio.gather cancels siblings but does not await
    their .close()).
    """

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
            # First two finish quickly; failing one loses the race.
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

        # Post-conditions:
        # 1. _size must equal 0 (all reservations released).
        assert pool._size == 0, (
            f"After partial-failure initialize, _size must be 0, got {pool._size}"
        )
        # 2. Every connection that got created must be EFFECTIVELY
        #    closed — not just method-called. initialize() does not
        #    flip _pool_released, so close_effective should be True.
        unclosed = [c for c in created if not c.close_effective]
        assert not unclosed, (
            f"Survivors of partial-failure gather must be effectively "
            f"closed; leaked: {[c.name for c in unclosed]}"
        )


class TestAcquireCancellationRestoresSize:
    """A caller cancelled while parked in acquire() must not leak the
    pool reservation. If a connection was pulled off the queue and
    handed to the cancelled task's get_task, it must either be
    returned to the pool or closed with _size decremented. Likewise
    for body-raised cancellation during yield cleanup.
    """

    async def test_cancelled_waiter_does_not_leak_size(self) -> None:
        pool, created = _make_pool_with_fake_cluster(max_size=1)
        # Pre-fill one connection so a second caller parks.
        await pool.initialize()

        # Take the one connection so the pool is at capacity.
        async def holder() -> None:
            async with pool.acquire():
                await asyncio.Event().wait()  # never release

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # Second caller will park waiting for the queue.
        async def waiter() -> None:
            async with pool.acquire():
                pass

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)

        # Cancel the waiter while it's parked on the queue.
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        # holder still holds its one slot; waiter never got one.
        assert pool._size == 1, (
            f"After cancelling a parked acquirer, _size must still be 1 "
            f"(the holder), got {pool._size}"
        )

        holder_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder_task
        await pool.close()

    async def test_reset_connection_failure_releases_reservation(self) -> None:
        """When the pool's cleanup path runs _reset_connection and it
        fails (e.g. ROLLBACK raises), the reservation must be released
        and the connection closed — never leaked.
        """

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
                # Simulate user-visible BEGIN so pool thinks there's a tx to roll back.
                c._in_transaction = True
                raise _BodyError()

        # After the cleanup, the reservation must be back and the conn
        # must have been effectively closed (not just method-called —
        # the pool previously set _pool_released before close(), making
        # close() a silent no-op and leaking the transport).
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
    """When pool.close() runs, every parked acquire() must return
    DqliteConnectionError promptly. The current clear()-then-wait
    pattern has a tiny window where close()'s set() can be erased,
    leaving waiters stalled until timeout.
    """

    async def test_close_wakes_parked_waiters(self) -> None:
        pool, _ = _make_pool_with_fake_cluster(max_size=1)
        await pool.initialize()

        # Hold the one connection.
        async def holder() -> None:
            async with pool.acquire():
                await asyncio.Event().wait()

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # Spawn N parked waiters.
        async def waiter() -> Any:
            try:
                async with pool.acquire():
                    return "got-conn"
            except DqliteConnectionError as e:
                return e

        N = 5
        waiters = [asyncio.create_task(waiter()) for _ in range(N)]
        await asyncio.sleep(0.05)

        # Close the pool. All waiters must return DqliteConnectionError
        # with the "Pool is closed" message. The MESSAGE is the real
        # invariant here: a waiter that hit the closed_event.clear()
        # race would wake via the per-poll timeout instead and raise
        # "Timed out waiting for a connection". We check the message
        # text, not elapsed time — CI latency is variable.
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
    """_reset_connection's ROLLBACK catch must not swallow programming
    errors or cancellation. Only transport-level categories should be
    treated as "connection is unhealthy, drop it"; anything else signals
    a bug or a structured-concurrency cancel that must propagate.
    """

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

        # Programming errors from ROLLBACK must not be silently logged
        # and converted into "drop the connection". A bare
        # `except BaseException` would swallow the AttributeError.
        with pytest.raises(AttributeError, match="stale attribute"):
            async with pool.acquire() as c:
                c._in_transaction = True
                raise _BodyDone()

        await pool.close()


class TestInitializeCleanupNarrowsException:
    """initialize() cleanup on partial failure must not swallow
    programming errors during conn.close(). A bare
    `suppress(BaseException)` masks refactor bugs and cancellation.
    """

    async def test_cleanup_programming_error_propagates(self) -> None:
        call_count = [0]

        class _CloseBoomConn(_FakeConn):
            async def close(self) -> None:
                raise AttributeError("close bug in survivor")

        async def _flaky_connect(**kwargs: Any) -> _FakeConn:
            n = call_count[0]
            call_count[0] += 1
            if n == 1:
                # Second connect fails → cleanup path runs on the first.
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

        # The cleanup loop previously wrapped conn.close() in
        # `suppress(BaseException)`, masking an AttributeError in a
        # survivor's close. After narrowing, the programming error
        # propagates out of initialize().
        with pytest.raises(AttributeError, match="close bug"):
            await pool.initialize()
