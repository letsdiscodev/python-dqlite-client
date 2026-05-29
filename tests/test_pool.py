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
        """bool / inf / nan are rejected, not coerced (True->1.0, inf disabling
        the deadline)."""
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
            await pool.initialize()

        assert create_count == 2  # min_size, not 2*min_size

        await pool.close()

    async def test_concurrent_initialize_creates_only_min_size_connections(self) -> None:
        """Concurrent initialize() calls must not create 2*min_size connections."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], min_size=2, max_size=5)

        create_count = 0

        async def mock_connect(**kwargs):
            nonlocal create_count
            create_count += 1
            await asyncio.sleep(0)  # yield to interleave
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
        """A failed initialize() must be retryable, not permanently stuck."""
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
            with pytest.raises(OSError):
                await pool.initialize()

            await pool.initialize()

        assert call_count == 2
        assert pool._size == 1

        await pool.close()

    async def test_acquire_cancels_first_task_if_second_create_task_raises(self) -> None:
        """If the second ``create_task`` in ``acquire()`` raises, the first task
        must be cancelled, not orphaned."""
        import contextlib as _contextlib

        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        # Hold the slot so the next acquire enters the queue-wait branch.
        async with pool.acquire():
            real_create_task = asyncio.create_task
            call_count = 0

            def flaky_create_task(coro, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    # Close the unscheduled coroutine to avoid a
                    # "coroutine was never awaited" warning.
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

        # Let any orphaned task surface warnings.
        with _contextlib.suppress(asyncio.TimeoutError):
            async with asyncio.timeout(0.05):
                await asyncio.sleep(1)

        await pool.close()

    async def test_acquire_logs_drain_idle_when_stale_conn_detected(self, caplog) -> None:
        """A stale-conn-triggered ``_drain_idle`` emits a DEBUG log."""
        import logging as _logging

        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=2)

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

            stale_conn.is_connected = False

            caplog.set_level(_logging.DEBUG, logger="dqliteclient.pool")
            async with pool.acquire():
                pass

        messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.pool"]
        assert any("drain-idle triggered by stale" in m for m in messages)

        await pool.close()

    async def test_pool_emits_debug_logs_for_lifecycle_events(self, caplog) -> None:
        """DEBUG traces initialize start/end, the at-capacity park, and close."""
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

            # Hold the slot and queue a second acquire to hit the capacity park.
            async with pool.acquire():
                waiter = asyncio.create_task(pool.acquire().__aenter__())
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
        """If a publish-step error escapes, the reservation is released so a
        retry starts from zero rather than climbing against a stale counter."""
        pool = ConnectionPool(["localhost:9001"], min_size=2, max_size=5)

        async def fake_create(**kwargs):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_conn.close = AsyncMock()
            return mock_conn

        with (
            patch.object(pool._cluster, "connect", side_effect=fake_create),
            patch.object(pool._pool, "put_nowait", side_effect=RuntimeError("synthetic")),
            pytest.raises(RuntimeError, match="synthetic"),
        ):
            await pool.initialize()

        assert pool._size == 0, f"_size should have been decremented back to 0, got {pool._size}"
        assert not pool._initialized

        await pool.close()

    async def test_cancellation_returns_healthy_connection_to_pool(self) -> None:
        """Cancelling a task holding a healthy conn returns it to the pool."""
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
                await asyncio.sleep(10)  # held until cancelled

        task = asyncio.create_task(hold_connection())
        await acquired.wait()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Cancel hit user code, not a protocol op, so the conn stays healthy.
        assert pool._size == initial_size
        assert pool._pool.qsize() == 1

        await pool.close()

    async def test_acquire_timeout_when_pool_exhausted(self) -> None:
        """acquire() must time out, not block forever, when the pool is exhausted."""
        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        # Must be explicitly falsy: a bare MagicMock attr auto-vivifies truthy
        # and would wrongly take _reset_connection's ROLLBACK branch.
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        async with pool.acquire():
            with pytest.raises(DqliteConnectionError, match="[Tt]imed out"):
                async with pool.acquire():
                    pass

    async def test_close_during_acquire_does_not_corrupt_size(self) -> None:
        """Closing the pool while conns are in use must not make _size negative."""
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

        await pool.close()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert pool._size >= 0, f"_size went negative: {pool._size}"

    async def test_acquire_recovers_after_connection_failure(self) -> None:
        """A waiter blocked on the queue must recover when capacity frees up."""
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
            holder_acquired = asyncio.Event()
            holder_release = asyncio.Event()

            async def holder():
                async with pool.acquire() as conn1:
                    assert conn1 is mock_conn1
                    holder_acquired.set()
                    await holder_release.wait()
                    conn1.is_connected = False  # simulate a broken connection
                    raise DqliteConnectionError("connection lost")

            holder_task = asyncio.create_task(holder())
            await holder_acquired.wait()

            # Waiter must enter the queue.get() wait (_size == max_size, queue empty).
            waiter_done = asyncio.Event()
            waiter_result: MagicMock | None = None

            async def waiter():
                nonlocal waiter_result
                async with pool.acquire() as c:
                    waiter_result = c  # type: ignore[assignment]
                    waiter_done.set()

            waiter_task = asyncio.create_task(waiter())
            # Let the waiter reach the queue.get() wait.
            await asyncio.sleep(0.1)

            # Release the holder: connection breaks, _size decrements, and the
            # blocked waiter should wake and create a new connection.
            holder_release.set()
            with contextlib.suppress(DqliteConnectionError):
                await holder_task

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
        """Dead-conn replacement must not let _size exceed max_size, even though
        the reservation pattern releases the lock before the handshake."""
        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=2)

        dead_conn = MagicMock()
        dead_conn.is_connected = False
        dead_conn.connect = AsyncMock()
        dead_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=dead_conn):
            await pool.initialize()

        assert pool._size == 1

        # _size must not exceed max_size even transiently: creates run outside
        # the lock, so concurrent acquires could otherwise race past the cap.
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
        """When a connection breaks, all idle connections are drained."""
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

            with pytest.raises(DqliteConnectionError):
                async with pool.acquire() as acquired:
                    acquired.is_connected = False  # type: ignore[misc]
                    raise DqliteConnectionError("connection lost")

        # Idle conns are drained too: they likely point at the same dead server.
        idle_closed = sum(1 for c in conns[1:] if c.close.called)
        assert idle_closed == 2, (
            f"Expected 2 idle connections to be drained, but only {idle_closed} were closed"
        )
        assert pool._pool.qsize() == 0

        await pool.close()

    async def test_dead_connection_triggers_leader_rediscovery(self) -> None:
        """A dead connection is replaced via leader discovery, not reconnected."""
        pool = ConnectionPool(["localhost:9001"], max_size=1)

        dead_conn = MagicMock()
        dead_conn.is_connected = False
        dead_conn.connect = AsyncMock()
        dead_conn.close = AsyncMock()

        new_conn = MagicMock()
        new_conn.is_connected = True
        new_conn.connect = AsyncMock()
        new_conn.close = AsyncMock()
        new_conn.execute = AsyncMock(return_value=(1, 1))

        with patch.object(pool._cluster, "connect", return_value=dead_conn):
            await pool.initialize()

        with patch.object(pool._cluster, "connect", return_value=new_conn):
            async with pool.acquire() as conn:
                assert conn is new_conn

        # No stale reconnect of the dead conn.
        dead_conn.connect.assert_not_called()

    async def test_close_then_return_closes_connection(self) -> None:
        """Returning a connection to a closed pool closes it, not re-queues it."""
        pool = ConnectionPool(["localhost:9001"], max_size=2)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        async with pool.acquire():
            await pool.close()

        mock_conn.close.assert_called()

    async def test_close_returns_immediately_when_in_flight_held(self) -> None:
        """``close()`` does not wait for checked-out conns (close-on-return,
        matching Go's ``database/sql.DB.Close``)."""
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
                await asyncio.sleep(60)

        holder = asyncio.create_task(hold_connection())
        await asyncio.sleep(0)  # let holder enter acquire

        # close() returns even though `holder` is still inside acquire().
        await asyncio.wait_for(pool.close(), timeout=2.0)

        # _size > 0 confirms in-flight has not been drained.
        assert pool._size > 0, "close() must not block on checked-out conns; documented contract"

        holder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await holder

    async def test_user_exception_preserves_healthy_connection(self) -> None:
        pool = ConnectionPool(["localhost:9001"], max_size=2)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()
        mock_conn._in_transaction = False

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        assert pool._size == 1

        with pytest.raises(ValueError, match="application error"):
            async with pool.acquire():
                raise ValueError("application error")

        mock_conn.close.assert_not_called()
        assert pool._size == 1
        assert pool._pool.qsize() == 1

        await pool.close()

    async def test_broken_connection_discarded_on_exception(self) -> None:
        """A broken connection is discarded even if user code raised."""
        pool = ConnectionPool(["localhost:9001"], max_size=2)

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.connect = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        with pytest.raises(ValueError, match="user error"):
            async with pool.acquire() as conn:
                conn.is_connected = False  # type: ignore[misc]
                raise ValueError("user error")

        mock_conn.close.assert_called()
        assert pool._size == 0

        await pool.close()

    async def test_drain_idle_cancellation_does_not_inflate_size(self) -> None:
        """If _drain_idle() is cancelled mid conn.close(), the orphan's slot is
        released by a follow-up, so _size converges to the queue size."""
        import asyncio

        pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=5)
        pool._initialized = True

        # close() blocks on this event so we can drive the orphan to completion
        # deterministically after the cancel and assert the deferred release.
        unblock = asyncio.Event()

        for _ in range(3):
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            mock_conn._in_use = False
            mock_conn._in_transaction = False
            mock_conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
            mock_conn._check_in_use = MagicMock()

            async def slow_close() -> None:
                await unblock.wait()

            mock_conn.close = slow_close
            await pool._pool.put(mock_conn)
            pool._size += 1

        assert pool._size == 3
        assert pool._pool.qsize() == 3

        task = asyncio.create_task(pool._drain_idle())
        await asyncio.sleep(0.01)  # let the first conn.close() begin
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Release the orphan close()s; the follow-ups then decrement _size.
        unblock.set()
        for _ in range(50):
            remaining = pool._pool.qsize()
            if pool._size == remaining:
                break
            await asyncio.sleep(0.01)

        remaining = pool._pool.qsize()
        assert pool._size == remaining, (
            f"_size ({pool._size}) != queue size ({remaining}). "
            f"After orphan drains finish, deferred slot releases should "
            f"have converged the count."
        )

    async def test_pool_close_cancellation_does_not_inflate_size(self) -> None:
        """If pool.close() is cancelled during conn.close(), _size still decrements."""
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

        assert pool._size >= 0
        assert pool._size < 2, (
            f"_size should have decremented, got {pool._size}. "
            f"CancelledError during close() skipped _size decrement."
        )

    async def test_pool_has_no_in_use_set(self) -> None:
        pool = ConnectionPool(["localhost:9001"])
        assert not hasattr(pool, "_in_use"), (
            "Pool._in_use set should have been removed — it was dead code (written but never read)"
        )

    async def test_release_rolls_back_open_transaction(self) -> None:
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

        # Leave _in_transaction=True on exit (e.g. a bug or raw BEGIN).
        async with pool.acquire() as conn:
            conn._in_transaction = True

        assert "ROLLBACK" in call_log, (
            f"Pool should issue ROLLBACK for dirty connections, calls: {call_log}"
        )
        assert not mock_conn._in_transaction

        await pool.close()

    async def test_release_destroys_connection_if_rollback_fails(self) -> None:
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
                # Transport-level error: _reset_connection only narrows to
                # these; a bare Exception would propagate as a bug.
                raise DqliteConnectionError("connection lost")
            return (0, 0)

        mock_conn.execute = failing_execute

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        initial_size = pool._size

        async with pool.acquire() as conn:
            conn._in_transaction = True

        mock_conn.close.assert_called()
        assert pool._size == initial_size - 1

        await pool.close()

    async def test_reset_connection_skips_rollback_on_dead_socket(self) -> None:
        """On a closing/EOF transport, _reset_connection returns False without a
        read timeout; otherwise every release on a half-closed socket costs
        self._timeout seconds."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=10.0)

        conn = DqliteConnection("127.0.0.1:9001")
        conn._protocol = MagicMock()
        conn._db_id = 1
        conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        conn._in_transaction = True

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
        """A waiter wakes promptly on release, not on the 500 ms poll cadence."""
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
        """A task parked in acquire() wakes quickly on pool.close(), not at its
        timeout."""
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
        """initialize() connects min_size in parallel: 5 conns at 0.1s each
        finish in ~0.1s, not ~0.5s serial."""
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
        """_reset_connection must let CancelledError propagate (the ROLLBACK
        catch is transport-only), not convert it to ``return False``."""
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
        """Using a connection after it returns to the pool raises InterfaceError."""
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

        escaped = None
        async with pool.acquire() as conn:
            escaped = conn
            await conn.execute("SELECT 1")

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

        async with pool.acquire() as conn:
            conn._in_transaction = True

        # ROLLBACK succeeded and the conn is back in the pool (not destroyed).
        assert not real_conn._in_transaction
        assert pool._pool.qsize() == 1
        assert pool._size == 1

        await pool.close()

    async def test_escaped_reference_works_when_reacquired(self) -> None:
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

        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

        # Second acquire reuses the same conn from the pool.
        async with pool.acquire() as conn:
            await conn.execute("SELECT 2")

        await pool.close()

    async def test_standalone_connection_not_affected_by_pool_guard(self) -> None:
        """A standalone DqliteConnection is unaffected by the pool guard."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        conn = DqliteConnection("127.0.0.1:9001")
        conn._protocol = MagicMock()
        conn._db_id = 1
        conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        conn._protocol.exec_sql = AsyncMock(return_value=(0, 1))

        await conn.execute("SELECT 1")
        await conn.execute("SELECT 2")  # _pool_released stays False, no guard


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
    """``_socket_looks_dead`` tolerates missing transport attrs (returns False
    so ROLLBACK is still tried) but must not swallow unexpected exceptions."""

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

        transport = object()
        reader = object()
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
    """``_drain_idle`` must propagate CancelledError (while still releasing the
    reservation) but absorb + log a non-cancellation close() failure so drain
    finishes the rest."""

    async def test_cancelled_error_in_close_propagates(self) -> None:
        """A CancelledError from close() must escape ``_drain_idle`` (the
        shielded per-conn close repackages it argless, but it still propagates)."""
        pool = ConnectionPool(["localhost:9001"])

        bad_conn = MagicMock()

        async def cancel_close() -> None:
            raise asyncio.CancelledError("outer timeout")

        bad_conn.close = cancel_close
        await pool._pool.put(bad_conn)
        pool._size = 1

        with pytest.raises(asyncio.CancelledError):
            await pool._drain_idle()

        # Reservation must still be released so the pool stays recoverable.
        assert pool._size == 0

    async def test_ordinary_exception_is_absorbed_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An OSError from close() is absorbed (DEBUG-logged) so drain continues."""
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
        # Log uses exc_info=True, so the message lives on the exception tuple.
        assert any(
            rec.exc_info is not None and "boom" in str(rec.exc_info[1]) for rec in caplog.records
        )


class TestAcquireCancellationPreservesCapacity:
    """Cancelling a coroutine parked in ``acquire()``'s ``asyncio.wait`` must not
    leak capacity: the abandoned ``get_task`` could win a later ``put()`` race,
    so it must be cancelled or its conn reclaimed."""

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
                    await holder_released.wait()

            async def waiter() -> None:
                async with pool.acquire():
                    pass

            holder_task = asyncio.create_task(holder())
            await waiter_parked.wait()

            waiter_task = asyncio.create_task(waiter())
            # Let waiter() park in asyncio.wait.
            await asyncio.sleep(0.05)

            waiter_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter_task

            # The post-cancel put() must not be stolen by the abandoned get_task.
            holder_released.set()
            await holder_task

            # If the abandoned task had captured the conn, this acquire times out.
            async with asyncio.timeout(1.0):
                async with pool.acquire() as conn:
                    assert conn is mock_connection

            assert pool._size == 1

        await pool.close()

    async def test_repeated_cancel_release_keeps_size_consistent(
        self,
        mock_connection: MagicMock,
    ) -> None:
        """After any cancel/release cycle ``_size`` equals the reachable conn
        count. Fuzzed over 100 rounds (kept light for CI)."""
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

                assert pool._size == 1

        await pool.close()


class TestReleaseCancellationPreservesReservation:
    """``_release`` must free the reservation even if CancelledError lands during
    ``_reset_connection`` or ``conn.close()`` — a per-branch shield on only the
    release call leaks ``_size`` when the cancel hits the preceding I/O."""

    async def test_cancel_during_reset_does_not_leak_reservation(self) -> None:
        """Path B: the shielded success-branch release runs ``_release`` to
        completion in the background even when an outer cancel lands at the
        release await."""
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

        rollback_started = asyncio.Event()
        unblock_rollback = asyncio.Event()

        async def gated_rollback(sql: str, params=None) -> tuple[int, int]:
            if "ROLLBACK" in sql:
                rollback_started.set()
                # Complete the rollback only after the outer cancel has run.
                await unblock_rollback.wait()
            return (0, 0)

        mock_conn.execute = gated_rollback

        with patch.object(pool._cluster, "connect", return_value=mock_conn):
            await pool.initialize()

        starting_size = pool._size

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(0.05):
                async with pool.acquire() as conn:
                    conn._in_transaction = True

        # Rollback beginning proves the shield kept _release running across the cancel.
        await asyncio.wait_for(rollback_started.wait(), timeout=1.0)
        unblock_rollback.set()
        # Let the shielded _release finish and re-queue the conn.
        for _ in range(50):
            if pool._pool.qsize() == 1:
                break
            await asyncio.sleep(0.01)

        # No leak: the slot stays accounted for by the re-queued idle conn
        # (the reservation transfers with it), so _size holds at starting_size.
        assert pool._size == starting_size, (
            f"reservation accounting mismatch: expected {starting_size}, got {pool._size}"
        )
        assert pool._pool.qsize() == 1, (
            f"conn should be returned to queue after successful reset; qsize={pool._pool.qsize()}"
        )

        await pool.close()

    async def test_cancel_during_pool_closed_close_does_not_leak(self) -> None:
        """Path A (pool closed): a cancel during the awaited ``conn.close()``
        must still release the reservation."""
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

        # Closed so _release takes path A.
        pool._closed = True
        pool._size = 1

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(0.05):
                await pool._release(mock_conn)

        assert pool._size == 0, f"reservation leaked on closed-pool path: got {pool._size}"

    async def test_cancel_during_queuefull_close_does_not_leak(self) -> None:
        """Path C (queue full): a cancel during the closing ``conn.close()``
        must still release the reservation."""
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

        # Queue already full (_size=2) so _release hits QueueFull.
        filler = MagicMock()
        await pool._pool.put(filler)
        pool._size = 2

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(0.05):
                await pool._release(mock_conn)

        assert pool._size == 1, f"reservation leaked on QueueFull path: got {pool._size}"
