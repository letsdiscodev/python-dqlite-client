"""Tests for connection pooling."""

import asyncio
import contextlib
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class TestConnectionPool:
    def test_min_size_greater_than_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="min_size.*must not exceed.*max_size"):
            ConnectionPool(["localhost:9001"], min_size=5, max_size=2)

    def test_negative_min_size_raises(self) -> None:
        with pytest.raises(ValueError, match="min_size.*non-negative"):
            ConnectionPool(["localhost:9001"], min_size=-1)

    def test_zero_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size.*at least 1"):
            ConnectionPool(["localhost:9001"], max_size=0)

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be a positive finite number"):
            ConnectionPool(["localhost:9001"], timeout=0)

    @pytest.mark.parametrize("bad", [True, float("inf"), float("nan")])
    def test_non_finite_or_bool_timeout_rejected(self, bad) -> None:
        """Cluster and pool timeouts share the same validator as
        ``DqliteConnection.__init__``: bool / inf / nan are rejected
        rather than silently coerced (``True`` becoming 1.0, ``inf``
        disabling the deadline).
        """
        with pytest.raises(ValueError, match="timeout must be a positive finite number"):
            ConnectionPool(["localhost:9001"], timeout=bad)

    def test_non_numeric_timeout_rejected(self) -> None:
        with pytest.raises(TypeError, match="timeout must be a number"):
            ConnectionPool(["localhost:9001"], timeout="10")  # type: ignore[arg-type]

    def test_init(self) -> None:
        pool = ConnectionPool(
            ["localhost:9001", "localhost:9002"],
            database="test",
            min_size=2,
            max_size=5,
        )
        assert pool._min_size == 2
        assert pool._max_size == 5

    async def test_close_empty_pool(self) -> None:
        pool = ConnectionPool(["localhost:9001"])
        await pool.close()
        assert pool._closed

    async def test_acquire_when_closed(self) -> None:
        pool = ConnectionPool(["localhost:9001"])
        pool._closed = True

        with pytest.raises(DqliteConnectionError, match="Pool is closed"):
            async with pool.acquire():
                pass

    async def test_double_initialize_is_idempotent(self) -> None:
        """Calling initialize() twice should not create double connections."""
        pool = ConnectionPool(["localhost:9001"], min_size=2, max_size=5)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()

        create_count = 0

        async def mock_connect(**kwargs):
            nonlocal create_count
            create_count += 1
            return mock_conn

        with patch.object(pool._cluster, "connect", side_effect=mock_connect):
            await pool.initialize()
            await pool.initialize()  # Second call should be no-op

        assert create_count == 2  # Only 2 (min_size), not 4

        await pool.close()

    async def test_concurrent_initialize_creates_only_min_size_connections(self) -> None:
        """Concurrent initialize() calls must not create 2*min_size connections."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], min_size=2, max_size=5)

        create_count = 0

        async def mock_connect(**kwargs):
            nonlocal create_count
            create_count += 1
            await asyncio.sleep(0)  # Yield to allow interleaving
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn

        with patch.object(pool._cluster, "connect", side_effect=mock_connect):
            await asyncio.gather(pool.initialize(), pool.initialize())

        assert create_count == 2, (
            f"Expected 2 connections (min_size), got {create_count}. "
            f"Concurrent initialize() created duplicate connections."
        )
        assert pool._size == 2

        await pool.close()

    async def test_initialize_retryable_after_failure(self) -> None:
        """If initialize() fails, it should be retryable (not permanently stuck)."""
        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=5)

        call_count = 0

        async def fail_then_succeed(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Connection refused")
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn

        with patch.object(pool._cluster, "connect", side_effect=fail_then_succeed):
            # First init fails
            with pytest.raises(OSError):
                await pool.initialize()

            # Second init should work (not be a no-op)
            await pool.initialize()

        assert call_count == 2  # First failed, second succeeded
        assert pool._size == 1

        await pool.close()

    async def test_acquire_cancels_first_task_if_second_create_task_raises(self) -> None:
        """If ``asyncio.create_task`` raises between the two task
        creations in ``acquire()`` (e.g. an outer ``CancelledError``
        firing on a coroutine switch), the first task must not be
        orphaned. The BaseException handler must cancel any task it
        managed to create.
        """
        import contextlib as _contextlib

        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        # Hold the single slot so acquire enters the queue-wait branch.
        async with pool.acquire():
            real_create_task = asyncio.create_task
            call_count = 0

            def flaky_create_task(coro, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    # Close the coroutine we won't schedule so we don't
                    # leak a "coroutine was never awaited" warning.
                    coro.close()
                    raise asyncio.CancelledError("synthetic interleave cancel")
                return real_create_task(coro, **kwargs)

            with (
                patch("dqliteclient.pool.asyncio.create_task", side_effect=flaky_create_task),
                pytest.raises(asyncio.CancelledError, match="synthetic interleave cancel"),
            ):
                async with asyncio.timeout(0.5):
                    async with pool.acquire():
                        pass

        # Give the loop a tick for any orphaned task to surface warnings.
        with _contextlib.suppress(asyncio.TimeoutError):
            async with asyncio.timeout(0.05):
                await asyncio.sleep(1)

        await pool.close()

    async def test_acquire_logs_drain_idle_when_stale_conn_detected(self, caplog) -> None:
        """A ``_drain_idle`` call triggered by a detected-stale
        connection emits a DEBUG log so the pool-wide replacement is
        visible to operators."""
        import logging as _logging

        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=2)

        # First connection reports healthy at initialize() time so it
        # ends up in the idle queue. Second connection (after drain +
        # fresh create) is the healthy replacement returned to acquire.
        stale_conn = MagicMock()
        stale_conn.is_connected = True
        stale_conn.connect = AsyncMock()
        stale_conn.close = AsyncMock()
        stale_conn._in_transaction = False
        fresh_conn = MagicMock()
        fresh_conn.is_connected = True
        fresh_conn.connect = AsyncMock()
        fresh_conn.close = AsyncMock()
        fresh_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", side_effect=[stale_conn, fresh_conn]):
            await pool.initialize()

            # Flip stale_conn to "not connected" so the next acquire
            # observes a dead conn and triggers drain.
            stale_conn.is_connected = False

            caplog.set_level(_logging.DEBUG, logger="dqliteclient.pool")
            async with pool.acquire():
                pass

        messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.pool"]
        assert any("drain-idle triggered by stale" in m for m in messages)

        await pool.close()

    async def test_pool_emits_debug_logs_for_lifecycle_events(self, caplog) -> None:
        """DEBUG traces the key pool state-change events so an operator
        can reconstruct pool behaviour from logs without adding
        bespoke instrumentation. Covers initialize start/end, the
        at-capacity park entry in acquire(), and close.
        """
        import logging as _logging

        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False

        caplog.set_level(_logging.DEBUG, logger="dqliteclient.pool")

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

            # Drive at-capacity wait: hold the single slot, queue a second
            # acquire that will time out, observe the capacity-wait log.
            async with pool.acquire():
                waiter = asyncio.create_task(pool.acquire().__aenter__())
                # Give the waiter a chance to reach the at-capacity park.
                for _ in range(10):
                    await asyncio.sleep(0)
                waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await waiter

        await pool.close()

        messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.pool"]
        assert any("pool.initialize: requesting" in m for m in messages)
        assert any("pool.initialize:" in m and "ready" in m for m in messages)
        assert any("pool.acquire: at capacity" in m for m in messages)
        assert any("pool.close: draining" in m for m in messages)

    async def test_initialize_size_resets_if_put_raises(self) -> None:
        """``_size`` must stay consistent with the actual pool membership
        even if a non-_POOL_CLEANUP_EXCEPTIONS error escapes during the
        put-to-queue phase of initialize(). Today the decrement only
        runs inside the ``if failures:`` branch — so a raise from
        ``self._pool.put`` (or an outer ``CancelledError`` mid-loop)
        leaks the reservation and a subsequent ``initialize()`` retry
        climbs toward ``_max_size`` without ever creating real
        connections.
        """
        pool = ConnectionPool(["localhost:9001"], min_size=2, max_size=5)

        async def fake_create(**kwargs):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn

        with (
            patch.object(pool._cluster, "connect", side_effect=fake_create),
            patch.object(pool._pool, "put", side_effect=RuntimeError("synthetic")),
            pytest.raises(RuntimeError, match="synthetic"),
        ):
            await pool.initialize()

        # Reservation must be released so retries start from zero.
        assert pool._size == 0, f"_size should have been decremented back to 0, got {pool._size}"
        assert not pool._initialized

        await pool.close()

    async def test_cancellation_returns_healthy_connection_to_pool(self) -> None:
        """Cancelling a task holding a healthy connection should return it to the pool."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        initial_size = pool._size
        acquired = asyncio.Event()

        async def hold_connection():
            async with pool.acquire() as _:
                acquired.set()
                await asyncio.sleep(10)  # Hold forever — will be cancelled

        task = asyncio.create_task(hold_connection())
        await acquired.wait()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Connection is still healthy (CancelledError was in user code, not in
        # a protocol operation), so it should be returned to the pool
        assert pool._size == initial_size
        assert pool._pool.qsize() == 1

        await pool.close()

    async def test_acquire_timeout_when_pool_exhausted(self) -> None:
        """acquire() should timeout, not block forever, when pool is exhausted."""
        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        # Must be explicitly falsy: the pool's _reset_connection takes
        # the ROLLBACK branch when _in_transaction is truthy, and
        # MagicMock's attribute-auto-vivification makes bare
        # ``mock_conn._in_transaction`` a truthy MagicMock.
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        # Check out the only connection
        async with pool.acquire():
            # Try to acquire another - should timeout
            with pytest.raises(DqliteConnectionError, match="[Tt]imed out"):
                async with pool.acquire():
                    pass

    async def test_close_during_acquire_does_not_corrupt_size(self) -> None:
        """Closing the pool while connections are in-use must not make _size negative."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], max_size=2)

        conns = []
        for _ in range(2):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_conn.close = AsyncMock()
            conns.append(mock_conn)

        conn_iter = iter(conns)
        with patch.object(pool._cluster, "connect", side_effect=lambda **kw: next(conn_iter)):
            await pool.initialize()

        acquired = asyncio.Event()

        async def hold_connection():
            async with pool.acquire() as _conn:
                acquired.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(hold_connection())
        await acquired.wait()

        # Close pool while task holds a connection
        await pool.close()

        # Cancel the task — its cleanup path will try to decrement _size
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # _size must never go negative
        assert pool._size >= 0, f"_size went negative: {pool._size}"

    async def test_acquire_recovers_after_connection_failure(self) -> None:
        """A waiter blocked on the queue should recover when capacity frees up."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1, timeout=2.0)

        mock_conn1 = MagicMock()
        mock_conn1.is_connected = True
        mock_conn1.connect = AsyncMock()
        mock_conn1.close = AsyncMock()

        mock_conn2 = MagicMock()
        mock_conn2.is_connected = True
        mock_conn2.connect = AsyncMock()
        mock_conn2.close = AsyncMock()

        conns = iter([mock_conn1, mock_conn2])

        with patch.object(pool._cluster, "connect", side_effect=lambda **kw: next(conns)):
            # Fill the pool to max_size and keep the connection checked out
            holder_acquired = asyncio.Event()
            holder_release = asyncio.Event()

            async def holder():
                async with pool.acquire() as conn1:
                    assert conn1 is mock_conn1
                    holder_acquired.set()
                    await holder_release.wait()
                    # Simulate a real connection failure (broken connection)
                    conn1.is_connected = False
                    raise DqliteConnectionError("connection lost")

            holder_task = asyncio.create_task(holder())
            await holder_acquired.wait()

            # Now start the waiter — it MUST enter the queue.get() wait
            # because _size == max_size and the queue is empty
            waiter_done = asyncio.Event()
            waiter_result: MagicMock | None = None

            async def waiter():
                nonlocal waiter_result
                async with pool.acquire() as c:
                    waiter_result = c  # type: ignore[assignment]
                    waiter_done.set()

            waiter_task = asyncio.create_task(waiter())
            # Give the waiter time to pass get_nowait (empty) and the lock
            # check (_size == max_size) and enter the queue.get() wait
            await asyncio.sleep(0.1)

            # Now release the holder — connection breaks, _size decrements
            holder_release.set()
            with contextlib.suppress(DqliteConnectionError):
                await holder_task
            # At this point _size == 0. The waiter is blocked on queue.get().
            # With the fix, the waiter should wake up and create a new connection.

            try:
                await asyncio.wait_for(waiter_done.wait(), timeout=1.5)
            except TimeoutError:
                waiter_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await waiter_task
                pytest.fail(
                    "Waiter timed out — pool starvation: waiter stuck on queue "
                    "even though _size dropped below max_size"
                )

        assert waiter_result is mock_conn2
        await pool.close()

    async def test_dead_conn_replacement_respects_max_size(self) -> None:
        """Dead-connection replacement must not allow _size to exceed max_size.

        An earlier change introduced the reservation pattern: the
        lock is released before the TCP handshake. The invariant we
        assert here is the one that actually matters — that
        ``_size`` never exceeds ``_max_size`` at any observation
        point — rather than the specific implementation detail of
        "lock held during create".
        """
        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=2)

        dead_conn = MagicMock()
        dead_conn.is_connected = False
        dead_conn.connect = AsyncMock()
        dead_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=dead_conn):
            await pool.initialize()

        assert pool._size == 1

        # Track _size at every create call. It must never exceed
        # max_size, even transiently, because creates happen outside
        # the lock under the new pattern and concurrent acquires
        # would otherwise race past the cap.
        peak_size_during_create = 0
        original_create = pool._create_connection

        async def tracking_create():
            nonlocal peak_size_during_create
            peak_size_during_create = max(peak_size_during_create, pool._size)
            return await original_create()

        new_conn = MagicMock()
        new_conn.is_connected = True
        new_conn.connect = AsyncMock()
        new_conn.close = AsyncMock()
        new_conn._in_transaction = False

        pool._create_connection = tracking_create
        with patch.object(pool._cluster, "connect", return_value=new_conn):
            async with pool.acquire() as conn:
                assert conn is new_conn

        assert peak_size_during_create <= pool._max_size, (
            f"_size exceeded max_size during dead-conn replacement "
            f"(peak={peak_size_during_create}, max={pool._max_size})"
        )

        await pool.close()

    async def test_broken_connection_drains_idle_pool(self) -> None:
        """When a connection breaks, all idle connections should be drained."""
        pool = ConnectionPool(["localhost:9001"], min_size=3, max_size=5)

        conns = []
        for _ in range(3):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_conn.close = AsyncMock()
            conns.append(mock_conn)

        new_conn = MagicMock()
        new_conn.is_connected = True
        new_conn.connect = AsyncMock()
        new_conn.close = AsyncMock()

        all_conns = [*conns, new_conn]
        conn_iter = iter(all_conns)
        with patch.object(pool._cluster, "connect", side_effect=lambda **kw: next(conn_iter)):
            await pool.initialize()

            assert pool._size == 3
            assert pool._pool.qsize() == 3

            # Acquire a connection — the first one from the queue
            with pytest.raises(DqliteConnectionError):
                async with pool.acquire() as acquired:
                    # Mark as broken (simulates leader change / server error)
                    acquired.is_connected = False  # type: ignore[misc]
                    raise DqliteConnectionError("connection lost")

        # The broken connection was discarded. The 2 idle connections should
        # also have been drained (they likely point to the same dead server).
        idle_closed = sum(1 for c in conns[1:] if c.close.called)
        assert idle_closed == 2, (
            f"Expected 2 idle connections to be drained, but only {idle_closed} were closed"
        )
        assert pool._pool.qsize() == 0

        await pool.close()

    async def test_dead_connection_triggers_leader_rediscovery(self) -> None:
        """A dead connection should be replaced via leader discovery, not reconnected."""
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        dead_conn = MagicMock()
        dead_conn.is_connected = False  # Connection is dead
        dead_conn.connect = AsyncMock()
        dead_conn.close = AsyncMock()

        new_conn = MagicMock()
        new_conn.is_connected = True
        new_conn.connect = AsyncMock()
        new_conn.close = AsyncMock()
        new_conn.execute = AsyncMock(return_value=(1, 1))

        # Initialize with the connection that will go dead
        with patch.object(pool._cluster, "connect", return_value=dead_conn):
            await pool.initialize()

        # When acquiring, the dead conn should be discarded and a new one created
        with patch.object(pool._cluster, "connect", return_value=new_conn):
            async with pool.acquire() as conn:
                assert conn is new_conn  # Got a fresh connection, not the dead one

        # Dead connection should NOT have had connect() called (no stale reconnect)
        dead_conn.connect.assert_not_called()

    async def test_close_then_return_closes_connection(self) -> None:
        """Returning a connection to a closed pool should close it, not put it back."""
        pool = ConnectionPool(["localhost:9001"], max_size=2)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        # Acquire, then close pool, then release — connection should be closed
        async with pool.acquire():
            await pool.close()

        # The connection should have been closed on return (not put back in queue)
        mock_conn.close.assert_called()

    async def test_close_returns_immediately_when_in_flight_held(self) -> None:
        """Pin: ``close()`` does NOT wait for in-flight checked-out
        connections to be returned. Documented contract — operators
        expecting drain-completion semantics must cancel/await
        in-flight tasks first. Matches go-dqlite's pool, which uses
        Go's ``database/sql.DB.Close`` close-on-return model."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], max_size=2)
        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        async def hold_connection() -> None:
            async with pool.acquire():
                # Hold for a while; close() must not wait for this.
                await asyncio.sleep(60)

        holder = asyncio.create_task(hold_connection())
        await asyncio.sleep(0)  # let holder enter acquire

        # close() returns even though `holder` is still inside acquire().
        await asyncio.wait_for(pool.close(), timeout=2.0)

        # _size > 0 confirms in-flight has not been drained.
        assert pool._size > 0, "close() must not block on checked-out conns; documented contract"

        # Cancel the holder so the conn can drain via _release's _closed path.
        holder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await holder

    async def test_user_exception_preserves_healthy_connection(self) -> None:
        """A user-code exception should not destroy a healthy connection."""
        pool = ConnectionPool(["localhost:9001"], max_size=2)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        assert pool._size == 1

        # User code raises a non-connection error
        with pytest.raises(ValueError, match="application error"):
            async with pool.acquire():
                raise ValueError("application error")

        # Connection should NOT have been closed — it's healthy
        mock_conn.close.assert_not_called()
        # Pool size should be unchanged
        assert pool._size == 1
        # Connection should be back in the pool queue
        assert pool._pool.qsize() == 1

        await pool.close()

    async def test_broken_connection_discarded_on_exception(self) -> None:
        """A broken connection should be discarded even if user code raised."""
        pool = ConnectionPool(["localhost:9001"], max_size=2)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        # Simulate a connection that becomes broken during use
        with pytest.raises(ValueError, match="user error"):
            async with pool.acquire() as conn:
                conn.is_connected = False  # type: ignore[misc]
                raise ValueError("user error")

        # Broken connection SHOULD have been closed and discarded
        mock_conn.close.assert_called()
        assert pool._size == 0

        await pool.close()

    async def test_drain_idle_cancellation_does_not_inflate_size(self) -> None:
        """If _drain_idle() is cancelled during conn.close(), _size must still decrement."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=5)
        pool._initialized = True

        # Create mock connections with a slow close that can be cancelled
        for _ in range(3):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn._in_use = False
            mock_conn._in_transaction = False
            mock_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
            mock_conn._check_in_use = MagicMock()

            async def slow_close():
                await asyncio.sleep(10)

            mock_conn.close = slow_close
            await pool._pool.put(mock_conn)
            pool._size += 1

        assert pool._size == 3
        assert pool._pool.qsize() == 3

        # Start draining and cancel mid-way
        task = asyncio.create_task(pool._drain_idle())
        await asyncio.sleep(0.01)  # Let drain start (first conn.close() begins)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Connections that were dequeued but not decremented due to CancelledError
        # during close() cause _size inflation. The invariant is:
        # _size == number of connections still in the queue
        remaining = pool._pool.qsize()
        assert pool._size == remaining, (
            f"_size ({pool._size}) != queue size ({remaining}). "
            f"CancelledError during conn.close() skipped _size decrement."
        )

    async def test_pool_close_cancellation_does_not_inflate_size(self) -> None:
        """If pool.close() is cancelled during conn.close(), _size must still decrement."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=5)
        pool._initialized = True

        for _ in range(2):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn._in_use = False
            mock_conn._in_transaction = False
            mock_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
            mock_conn._check_in_use = MagicMock()

            async def slow_close():
                await asyncio.sleep(10)

            mock_conn.close = slow_close
            await pool._pool.put(mock_conn)
            pool._size += 1

        assert pool._size == 2

        task = asyncio.create_task(pool.close())
        await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # _size must reflect that connections were removed from queue
        assert pool._size >= 0
        # At least the first connection should have been decremented
        assert pool._size < 2, (
            f"_size should have decremented, got {pool._size}. "
            f"CancelledError during close() skipped _size decrement."
        )

    async def test_pool_has_no_in_use_set(self) -> None:
        """Pool should not maintain a dead _in_use set (removed as dead code)."""
        pool = ConnectionPool(["localhost:9001"])
        assert not hasattr(pool, "_in_use"), (
            "Pool._in_use set should have been removed — it was dead code (written but never read)"
        )

    async def test_release_rolls_back_open_transaction(self) -> None:
        """Returning a connection with _in_transaction=True must issue ROLLBACK."""
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._check_in_use = MagicMock()

        call_log: list[str] = []

        async def mock_execute(sql, params=None):
            call_log.append(sql)
            return (0, 0)

        mock_conn.execute = mock_execute

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        # Simulate: user enters transaction but the context manager exit
        # somehow leaves _in_transaction=True (e.g., a bug or raw BEGIN)
        async with pool.acquire() as conn:
            conn._in_transaction = True

        # The pool should have issued ROLLBACK before returning to queue
        assert "ROLLBACK" in call_log, (
            f"Pool should issue ROLLBACK for dirty connections, calls: {call_log}"
        )
        # And the flag should be reset
        assert not mock_conn._in_transaction

        await pool.close()

    async def test_release_destroys_connection_if_rollback_fails(self) -> None:
        """If ROLLBACK fails on dirty release, connection must be destroyed."""
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._check_in_use = MagicMock()

        async def failing_execute(sql, params=None):
            if "ROLLBACK" in sql:
                # Use a transport-level error: _reset_connection narrows
                # to (OSError, TimeoutError, DqliteConnectionError,
                # ProtocolError, OperationalError) so programming-error
                # categories like bare Exception propagate as bugs.
                raise DqliteConnectionError("connection lost")
            return (0, 0)

        mock_conn.execute = failing_execute

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        initial_size = pool._size

        async with pool.acquire() as conn:
            conn._in_transaction = True

        # ROLLBACK failed, so connection should be destroyed
        mock_conn.close.assert_called()
        assert pool._size == initial_size - 1

        await pool.close()

    async def test_reset_connection_skips_rollback_on_dead_socket(self) -> None:
        """If the transport is already closing / EOF, _reset_connection must
        return False without waiting for a read timeout. Otherwise every
        release through a half-closed socket costs self._timeout seconds.
        """
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=10.0)

        conn = DqliteConnection("127.0.0.1:9001")
        conn._protocol = MagicMock()
        conn._db_id = 1
        conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        conn._in_transaction = True

        # Reader has seen EOF / transport is closing.
        dead_reader = MagicMock()
        dead_reader.at_eof = MagicMock(return_value=True)
        dead_transport = MagicMock()
        dead_transport.is_closing = MagicMock(return_value=True)
        dead_writer = MagicMock()
        dead_writer.transport = dead_transport
        conn._protocol._reader = dead_reader
        conn._protocol._writer = dead_writer

        exec_called = False

        async def track_exec(*args, **kwargs):
            nonlocal exec_called
            exec_called = True
            # Never resolves — we'd pay self._timeout if reached.
            await asyncio.sleep(1000)
            return (0, 0)

        conn._protocol.exec_sql = track_exec
        conn._protocol.close = MagicMock()
        conn._protocol.wait_closed = AsyncMock()

        result = await asyncio.wait_for(pool._reset_connection(conn), timeout=1.0)
        assert result is False
        assert not exec_called, "ROLLBACK must not be sent on a dead socket"

    async def test_acquire_wakes_on_release_without_polling_delay(self) -> None:
        """When a held connection is released, a waiter must wake up
        promptly — not sit on the 500 ms polling cadence.
        """
        import asyncio
        import time

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=5.0)

        mock_conn = MagicMock(spec=DqliteConnection)
        mock_conn.is_connected = True
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._pool_released = False
        mock_conn._address = "localhost:9001"
        mock_conn._check_in_use = MagicMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

            holder_acquired = asyncio.Event()
            release_holder = asyncio.Event()

            async def hold_connection() -> None:
                async with pool.acquire():
                    holder_acquired.set()
                    await release_holder.wait()

            holder = asyncio.create_task(hold_connection())
            await holder_acquired.wait()

            async def grab() -> float:
                t0 = time.monotonic()
                async with pool.acquire():
                    return time.monotonic() - t0

            waiter = asyncio.create_task(grab())
            await asyncio.sleep(0.05)
            release_holder.set()
            elapsed = await asyncio.wait_for(waiter, timeout=1.0)
            await holder

            await pool.close()

        assert elapsed < 0.2, f"waiter should wake immediately on release; took {elapsed:.3f}s"

    async def test_close_wakes_waiter_promptly(self) -> None:
        """A task blocked in acquire() waiting for a connection must be
        woken quickly when pool.close() is called — not sit on the queue
        until its timeout expires.
        """
        import asyncio
        import time

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=5.0)

        mock_conn = MagicMock(spec=DqliteConnection)
        mock_conn.is_connected = True
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._pool_released = False
        mock_conn._address = "localhost:9001"
        mock_conn._check_in_use = MagicMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

            holder_acquired = asyncio.Event()
            release_holder = asyncio.Event()

            async def hold_connection() -> None:
                async with pool.acquire():
                    holder_acquired.set()
                    await release_holder.wait()

            holder = asyncio.create_task(hold_connection())
            await holder_acquired.wait()

            async def try_acquire() -> BaseException | None:
                try:
                    async with pool.acquire():
                        return None
                except BaseException as e:  # noqa: BLE001
                    return e

            waiter = asyncio.create_task(try_acquire())
            await asyncio.sleep(0.05)

            t0 = time.monotonic()
            await pool.close()
            err = await asyncio.wait_for(waiter, timeout=1.0)
            elapsed = time.monotonic() - t0

            release_holder.set()
            await holder

        assert isinstance(err, DqliteConnectionError)
        assert elapsed < 0.3, (
            f"acquire() should wake within ~100ms of pool.close(); took {elapsed:.3f}s"
        )

    async def test_initialize_creates_min_size_in_parallel(self) -> None:
        """initialize() should connect min_size in parallel, not sequentially.
        With min_size=5 and per-connect latency 0.1s, serial init takes ~0.5s;
        parallel should finish in ~0.1s.
        """
        import asyncio
        import time

        pool = ConnectionPool(["localhost:9001"], min_size=5, max_size=10)

        async def slow_connect(**kwargs):
            await asyncio.sleep(0.1)
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.close = AsyncMock()
            return mock_conn

        with patch.object(pool._cluster, "connect", side_effect=slow_connect):
            t0 = time.monotonic()
            await pool.initialize()
            elapsed = time.monotonic() - t0

        assert pool._size == 5
        assert elapsed < 0.3, (
            f"expected parallel init (~0.1s), got {elapsed:.3f}s "
            f"— looks like the connections were created sequentially"
        )

        await pool.close()

    async def test_pool_accepts_injected_cluster_client(self) -> None:
        """Callers must be able to share a ClusterClient across multiple pools."""
        from dqliteclient.cluster import ClusterClient

        shared_cluster = ClusterClient.from_addresses(["localhost:9001"])
        pool_a = ConnectionPool(cluster=shared_cluster, min_size=0, max_size=3)
        pool_b = ConnectionPool(cluster=shared_cluster, min_size=0, max_size=3)
        assert pool_a._cluster is shared_cluster
        assert pool_b._cluster is shared_cluster

    async def test_pool_accepts_injected_node_store(self) -> None:
        """Callers with a persistent NodeStore must be able to thread it in."""
        from dqliteclient.node_store import MemoryNodeStore

        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        pool = ConnectionPool(node_store=store, min_size=0, max_size=1)
        nodes = await pool._cluster._node_store.get_nodes()
        assert [n.address for n in nodes] == ["localhost:9001", "localhost:9002"]

    async def test_pool_requires_some_cluster_source(self) -> None:
        """Constructing with neither addresses nor cluster/node_store must raise."""
        with pytest.raises(ValueError, match="addresses.*cluster.*node_store"):
            ConnectionPool()

    async def test_pool_rejects_cluster_and_node_store_together(self) -> None:
        """Passing both cluster= and node_store= must raise — pick one."""
        from dqliteclient.cluster import ClusterClient
        from dqliteclient.node_store import MemoryNodeStore

        cluster = ClusterClient.from_addresses(["localhost:9001"])
        store = MemoryNodeStore(["localhost:9001"])
        with pytest.raises(ValueError, match="only one"):
            ConnectionPool(cluster=cluster, node_store=store)

    async def test_reset_connection_propagates_cancelled_error(self) -> None:
        """_reset_connection must let CancelledError propagate, not
        convert it into a ``return False`` (connection-drop). Catching
        CancelledError would silently eat structured-concurrency
        cancellations delivered from a parent ``asyncio.timeout()`` or
        sibling-task cancel. The ROLLBACK catch is scoped to transport
        errors only.
        """
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        conn = DqliteConnection("127.0.0.1:9001")
        conn._protocol = MagicMock()
        conn._db_id = 1
        conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        conn._in_transaction = True

        async def cancel_on_rollback(db_id, sql, params=None):
            if "ROLLBACK" in str(sql):
                raise asyncio.CancelledError()
            return (0, 0)

        conn._protocol.exec_sql = cancel_on_rollback
        conn._protocol.close = MagicMock()
        conn._protocol.wait_closed = AsyncMock()

        with pytest.raises(asyncio.CancelledError):
            await pool._reset_connection(conn)

    async def test_escaped_reference_rejected_after_release(self) -> None:
        """Using a connection after it's returned to the pool must raise InterfaceError."""
        import asyncio

        from dqliteclient.connection import DqliteConnection
        from dqliteclient.exceptions import InterfaceError

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        real_conn = DqliteConnection("127.0.0.1:9001")
        real_conn._protocol = MagicMock()
        real_conn._db_id = 1
        real_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        real_conn._protocol.exec_sql = AsyncMock(return_value=(0, 1))
        real_conn._protocol.query_sql = AsyncMock(return_value=(["id"], [[1]]))
        real_conn._protocol.close = MagicMock()
        real_conn._protocol.wait_closed = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=real_conn):
            await pool.initialize()

        # Stash a reference to the connection
        escaped = None
        async with pool.acquire() as conn:
            escaped = conn
            await conn.execute("SELECT 1")

        # Connection is now back in the pool — escaped reference must be rejected
        assert escaped is not None
        with pytest.raises(InterfaceError, match="returned to the pool"):
            await escaped.execute("SELECT 1")

        await pool.close()

    async def test_escaped_reference_rejected_after_exception(self) -> None:
        """Escaped reference must be rejected even when user code raises."""
        import asyncio

        from dqliteclient.connection import DqliteConnection
        from dqliteclient.exceptions import InterfaceError

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        real_conn = DqliteConnection("127.0.0.1:9001")
        real_conn._protocol = MagicMock()
        real_conn._db_id = 1
        real_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        real_conn._protocol.exec_sql = AsyncMock(return_value=(0, 1))
        real_conn._protocol.close = MagicMock()
        real_conn._protocol.wait_closed = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=real_conn):
            await pool.initialize()

        escaped = None
        with pytest.raises(ValueError, match="app error"):
            async with pool.acquire() as conn:
                escaped = conn
                raise ValueError("app error")

        assert escaped is not None
        with pytest.raises(InterfaceError, match="returned to the pool"):
            await escaped.execute("SELECT 1")

        await pool.close()

    async def test_pool_release_rolls_back_transaction_with_real_connection(self) -> None:
        """_reset_connection must be able to ROLLBACK before _pool_released is set."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        real_conn = DqliteConnection("127.0.0.1:9001")
        real_conn._protocol = MagicMock()
        real_conn._db_id = 1
        real_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        real_conn._protocol.exec_sql = AsyncMock(return_value=(0, 0))
        real_conn._protocol.close = MagicMock()
        real_conn._protocol.wait_closed = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=real_conn):
            await pool.initialize()

        # Simulate a connection with an open transaction returned to pool
        async with pool.acquire() as conn:
            conn._in_transaction = True

        # The pool should have issued ROLLBACK successfully (not destroyed the conn)
        assert not real_conn._in_transaction
        # Connection should be back in the pool (not destroyed)
        assert pool._pool.qsize() == 1
        assert pool._size == 1

        await pool.close()

    async def test_escaped_reference_works_when_reacquired(self) -> None:
        """A connection re-acquired from the pool must work normally."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        real_conn = DqliteConnection("127.0.0.1:9001")
        real_conn._protocol = MagicMock()
        real_conn._db_id = 1
        real_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        real_conn._protocol.exec_sql = AsyncMock(return_value=(0, 1))
        real_conn._protocol.close = MagicMock()
        real_conn._protocol.wait_closed = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=real_conn):
            await pool.initialize()

        # First acquire and release
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

        # Second acquire — same connection from pool must work
        async with pool.acquire() as conn:
            await conn.execute("SELECT 2")

        await pool.close()

    async def test_standalone_connection_not_affected_by_pool_guard(self) -> None:
        """A DqliteConnection used standalone (not from a pool) must not be affected."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        conn = DqliteConnection("127.0.0.1:9001")
        conn._protocol = MagicMock()
        conn._db_id = 1
        conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        conn._protocol.exec_sql = AsyncMock(return_value=(0, 1))

        # Standalone connection — no pool involved, should work fine
        await conn.execute("SELECT 1")
        await conn.execute("SELECT 2")  # No error — _pool_released is always False


class TestConnectionPoolIntegration:
    """Integration tests requiring mocked connections."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        conn = MagicMock()
        conn.is_connected = True
        conn.connect = AsyncMock()
        conn.close = AsyncMock()
        conn.execute = AsyncMock(return_value=(1, 1))
        conn.fetch = AsyncMock(return_value=[{"id": 1}])
        return conn

    async def test_execute_through_pool(self, mock_connection: MagicMock) -> None:
        pool = ConnectionPool(["localhost:9001"])

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()

            result = await pool.execute("INSERT INTO t VALUES (1)")
            assert result == (1, 1)

        await pool.close()

    async def test_fetch_through_pool(self, mock_connection: MagicMock) -> None:
        pool = ConnectionPool(["localhost:9001"])

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()

            result = await pool.fetch("SELECT * FROM t")
            assert result == [{"id": 1}]

        await pool.close()

    async def test_fetchone_through_pool(self, mock_connection: MagicMock) -> None:
        mock_connection.fetchone = AsyncMock(return_value={"id": 1})
        pool = ConnectionPool(["localhost:9001"])

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()
            result = await pool.fetchone("SELECT * FROM t LIMIT 1")
            assert result == {"id": 1}

        await pool.close()

    async def test_fetchall_through_pool(self, mock_connection: MagicMock) -> None:
        mock_connection.fetchall = AsyncMock(return_value=[[1, "a"], [2, "b"]])
        pool = ConnectionPool(["localhost:9001"])

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()
            result = await pool.fetchall("SELECT * FROM t")
            assert result == [[1, "a"], [2, "b"]]

        await pool.close()

    async def test_fetchval_through_pool(self, mock_connection: MagicMock) -> None:
        mock_connection.fetchval = AsyncMock(return_value=42)
        pool = ConnectionPool(["localhost:9001"])

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()
            result = await pool.fetchval("SELECT count(*) FROM t")
            assert result == 42

        await pool.close()


class TestSocketLooksDead:
    """Contract tests for the pool's ``_socket_looks_dead`` heuristic.

    The helper must tolerate mocked / missing transport attributes (returning
    ``False`` so ROLLBACK still gets attempted) but must NOT silently swallow
    arbitrary exceptions — an unexpected error is a programmer bug that should
    surface, not a signal of a dead socket.
    """

    def _make_conn(
        self,
        *,
        transport: object | None = None,
        reader: object | None = None,
    ) -> MagicMock:
        writer = MagicMock()
        writer.transport = transport
        protocol = MagicMock()
        protocol._writer = writer
        protocol._reader = reader
        conn = MagicMock()
        conn._protocol = protocol
        return conn

    def test_returns_true_when_protocol_is_none(self) -> None:
        from dqliteclient.pool import _socket_looks_dead

        conn = MagicMock()
        conn._protocol = None
        assert _socket_looks_dead(conn) is True

    def test_returns_false_when_transport_and_reader_both_alive(self) -> None:
        from dqliteclient.pool import _socket_looks_dead

        transport = MagicMock()
        transport.is_closing = MagicMock(return_value=False)
        reader = MagicMock()
        reader.at_eof = MagicMock(return_value=False)

        conn = self._make_conn(transport=transport, reader=reader)
        assert _socket_looks_dead(conn) is False

    def test_returns_true_when_transport_is_closing(self) -> None:
        from dqliteclient.pool import _socket_looks_dead

        transport = MagicMock()
        transport.is_closing = MagicMock(return_value=True)
        reader = MagicMock()
        reader.at_eof = MagicMock(return_value=False)

        conn = self._make_conn(transport=transport, reader=reader)
        assert _socket_looks_dead(conn) is True

    def test_returns_true_when_reader_at_eof(self) -> None:
        from dqliteclient.pool import _socket_looks_dead

        transport = MagicMock()
        transport.is_closing = MagicMock(return_value=False)
        reader = MagicMock()
        reader.at_eof = MagicMock(return_value=True)

        conn = self._make_conn(transport=transport, reader=reader)
        assert _socket_looks_dead(conn) is True

    def test_attribute_error_falls_through_as_alive(self) -> None:
        """Mocks missing ``is_closing`` / ``at_eof`` default to "assume alive"."""
        from dqliteclient.pool import _socket_looks_dead

        transport = object()  # no is_closing attribute
        reader = object()  # no at_eof attribute
        conn = self._make_conn(transport=transport, reader=reader)
        assert _socket_looks_dead(conn) is False

    def test_unexpected_exception_is_not_swallowed(self) -> None:
        """A ``ValueError`` from a broken mock must propagate, not be suppressed."""
        from dqliteclient.pool import _socket_looks_dead

        transport = MagicMock()
        transport.is_closing = MagicMock(side_effect=ValueError("broken mock"))
        reader = MagicMock()
        reader.at_eof = MagicMock(return_value=False)

        conn = self._make_conn(transport=transport, reader=reader)
        with pytest.raises(ValueError, match="broken mock"):
            _socket_looks_dead(conn)


class TestDrainIdleCancellation:
    """``_drain_idle`` must not swallow ``CancelledError`` or any other
    ``BaseException``. The reservation release on each iteration still runs,
    so cancellation is clean — but the cancel signal itself must propagate.

    A close() failure for a non-cancellation reason should be absorbed so
    drain can finish the remaining connections, but logged for diagnostics.
    """

    async def test_cancelled_error_in_close_propagates(self) -> None:
        """An async close() raising ``CancelledError`` must escape ``_drain_idle``.
        Previously the helper's ``except BaseException: pass`` swallowed it,
        breaking structured concurrency (``asyncio.timeout`` / ``TaskGroup``).

        Note: per-connection close is now shielded so the original
        cancel's message is lost through the Task boundary — asyncio
        repackages any CancelledError the shielded task raises into a
        fresh CancelledError with no args. The semantic we pin here is
        that CancelledError still propagates out of the drain (i.e. is
        not swallowed).
        """
        pool = ConnectionPool(["localhost:9001"])

        bad_conn = MagicMock()

        async def cancel_close() -> None:
            raise asyncio.CancelledError("outer timeout")

        bad_conn.close = cancel_close
        await pool._pool.put(bad_conn)
        pool._size = 1

        with pytest.raises(asyncio.CancelledError):
            await pool._drain_idle()

        # Reservation must still be released on the cancellation path so
        # the pool is recoverable after the cancel.
        assert pool._size == 0

    async def test_ordinary_exception_is_absorbed_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An ``OSError`` from close() should be absorbed so drain continues.
        Behaviour is logged at DEBUG for operator diagnostics.
        """
        import logging

        pool = ConnectionPool(["localhost:9001"])

        c1, c2 = MagicMock(), MagicMock()
        c1._address = "n1:9001"
        c2._address = "n2:9001"
        c1.close = AsyncMock(side_effect=OSError("boom"))
        c2.close = AsyncMock()
        await pool._pool.put(c1)
        await pool._pool.put(c2)
        pool._size = 2

        with caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"):
            await pool._drain_idle()

        c2.close.assert_awaited_once()
        assert pool._size == 0
        # The cleanup-path log uses exc_info=True so the exception
        # message lives on the record's exception tuple, not in the
        # plain message.
        assert any(
            rec.exc_info is not None and "boom" in str(rec.exc_info[1]) for rec in caplog.records
        )


class TestAcquireCancellationPreservesCapacity:
    """An external cancellation of a coroutine parked in ``acquire()``'s
    ``asyncio.wait(...)`` must not leak a connection or shrink effective
    pool capacity. The abandoned ``get_task`` could win a subsequent
    ``put()`` race; the fix must either cancel it or reclaim the
    connection it took.
    """

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        conn = MagicMock()
        conn.is_connected = True
        conn._in_transaction = False
        conn._pool_released = False
        conn.connect = AsyncMock()
        conn.close = AsyncMock()
        conn._protocol = MagicMock()
        conn._protocol._writer = MagicMock()
        conn._protocol._writer.transport = MagicMock()
        conn._protocol._writer.transport.is_closing = MagicMock(return_value=False)
        conn._protocol._reader = MagicMock()
        conn._protocol._reader.at_eof = MagicMock(return_value=False)
        return conn

    async def test_cancelled_waiter_does_not_wedge_pool(
        self,
        mock_connection: MagicMock,
    ) -> None:
        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=5.0)

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()

            holder_released = asyncio.Event()
            waiter_parked = asyncio.Event()

            async def holder() -> None:
                async with pool.acquire():
                    waiter_parked.set()
                    # Hold until the waiter has been cancelled AND we've
                    # been signalled to exit.
                    await holder_released.wait()

            async def waiter() -> None:
                async with pool.acquire():
                    pass

            holder_task = asyncio.create_task(holder())
            await waiter_parked.wait()

            waiter_task = asyncio.create_task(waiter())
            # Give waiter() time to enter acquire() and park in asyncio.wait.
            await asyncio.sleep(0.05)

            waiter_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter_task

            # Release the in-use connection; the post-cancel put() must
            # not be stolen by the abandoned get_task.
            holder_released.set()
            await holder_task

            # Pool must still accept a new acquire within a short window —
            # if the abandoned task captured the connection, this would
            # time out.
            async with asyncio.timeout(1.0):
                async with pool.acquire() as conn:
                    assert conn is mock_connection

            assert pool._size == 1

        await pool.close()

    async def test_repeated_cancel_release_keeps_size_consistent(
        self,
        mock_connection: MagicMock,
    ) -> None:
        """Invariant: after any cancel/release cycle the ``_size`` counter
        equals the number of connections actually reachable. Fuzz-test 100
        rounds (smaller than the issue file's 1000 — kept light for CI)."""
        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=5.0)

        with patch.object(pool._cluster, "connect", return_value=mock_connection):
            await pool.initialize()

            for _ in range(100):
                release_event = asyncio.Event()

                async def holder(event: asyncio.Event = release_event) -> None:
                    async with pool.acquire():
                        await event.wait()

                async def waiter() -> None:
                    async with pool.acquire():
                        pass

                h = asyncio.create_task(holder())
                await asyncio.sleep(0)
                w = asyncio.create_task(waiter())
                await asyncio.sleep(0.01)
                w.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await w
                release_event.set()
                await h

                # After every round the pool must have exactly one
                # reachable reservation — queued or returned-by-holder.
                assert pool._size == 1

        await pool.close()


class TestReleaseCancellationPreservesReservation:
    """``ConnectionPool._release`` must release the reservation even if
    ``CancelledError`` is delivered during ``_reset_connection`` (the
    ROLLBACK await) or during the inline ``conn.close()`` calls.

    The earlier per-branch ``asyncio.shield`` only protected the
    reservation-release call itself; cancellation arriving DURING the
    preceding I/O bypassed every subsequent line, including the shield,
    leaving ``_size`` permanently incremented.
    """

    async def test_cancel_during_reset_does_not_leak_reservation(self) -> None:
        """Path B (production-likely): the success branch of
        ``acquire()`` calls ``_release`` → ``_reset_connection`` →
        ``conn.execute("ROLLBACK")``. If an outer cancellation arrives
        at the ROLLBACK await, the reservation must still be released.
        """
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._check_in_use = MagicMock()
        mock_conn._pool_released = False

        async def slow_rollback(sql: str, params=None) -> tuple[int, int]:
            if "ROLLBACK" in sql:
                # Suspend long enough for the outer timeout to fire.
                await asyncio.sleep(10)
            return (0, 0)

        mock_conn.execute = slow_rollback

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        starting_size = pool._size

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(0.05):
                async with pool.acquire() as conn:
                    conn._in_transaction = True

        # The reservation MUST have been released even though
        # CancelledError fired during _reset_connection's ROLLBACK.
        assert pool._size == starting_size - 1, (
            f"reservation leaked: expected {starting_size - 1}, got {pool._size}"
        )
        # _pool_released is intentionally NOT asserted: cancellation
        # arriving mid-close leaves the conn an orphan with the flag
        # still False, so a holder of the ref can still close it via
        # ``DqliteConnection.close()``. The pool's only obligation is
        # the reservation count.

        await pool.close()

    async def test_cancel_during_pool_closed_close_does_not_leak(self) -> None:
        """Path A: pool already closed; ``conn.close()`` is awaited.
        If the close suspends and an outer cancel fires, the
        reservation must still be released.
        """
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._check_in_use = MagicMock()
        mock_conn._pool_released = False

        async def slow_close() -> None:
            await asyncio.sleep(10)

        mock_conn.close = slow_close

        # Pre-mark pool as closed so _release takes path A.
        pool._closed = True
        pool._size = 1

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(0.05):
                await pool._release(mock_conn)

        assert pool._size == 0, f"reservation leaked on closed-pool path: got {pool._size}"

    async def test_cancel_during_queuefull_close_does_not_leak(self) -> None:
        """Path C: queue full; the connection must close. If the
        close suspends and an outer cancel fires, the reservation must
        still be released.
        """
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn._in_transaction = False
        mock_conn._in_use = False
        mock_conn._bound_loop_ref = None
        mock_conn._check_in_use = MagicMock()
        mock_conn._pool_released = False

        async def slow_close() -> None:
            await asyncio.sleep(10)

        mock_conn.close = slow_close

        # Reservation reflects two inflight connections; queue is full
        # with one already so _release will hit QueueFull.
        filler = MagicMock()
        await pool._pool.put(filler)
        pool._size = 2

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(0.05):
                await pool._release(mock_conn)

        # The mock_conn's reservation must have been released; the
        # filler stays queued.
        assert pool._size == 1, f"reservation leaked on QueueFull path: got {pool._size}"
