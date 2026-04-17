"""Tests for connection pooling."""

import contextlib
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
        with pytest.raises(ValueError, match="timeout must be positive"):
            ConnectionPool(["localhost:9001"], timeout=0)

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
                    waiter_result = c
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
        """Dead-connection replacement must not allow _size to exceed max_size."""
        pool = ConnectionPool(["localhost:9001"], min_size=1, max_size=2)

        dead_conn = MagicMock()
        dead_conn.is_connected = False
        dead_conn.connect = AsyncMock()
        dead_conn.close = AsyncMock()

        with patch.object(pool._cluster, "connect", return_value=dead_conn):
            await pool.initialize()

        assert pool._size == 1

        # Track whether the lock is held during dead-conn replacement.
        # If _create_connection is called while the lock is NOT held,
        # another coroutine could race and exceed max_size.
        lock_was_held_during_create = False
        original_create = pool._create_connection

        async def tracking_create():
            nonlocal lock_was_held_during_create
            if pool._lock.locked():
                lock_was_held_during_create = True
            return await original_create()

        new_conn = MagicMock()
        new_conn.is_connected = True
        new_conn.connect = AsyncMock()
        new_conn.close = AsyncMock()

        pool._create_connection = tracking_create
        with patch.object(pool._cluster, "connect", return_value=new_conn):
            async with pool.acquire() as conn:
                assert conn is new_conn

        assert lock_was_held_during_create, (
            "Dead-connection replacement must hold the lock to prevent "
            "_size race with concurrent acquire() calls"
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
                    acquired.is_connected = False
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
                conn.is_connected = False  # Mark as broken (simulates invalidation)
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
            mock_conn._bound_loop = asyncio.get_running_loop()
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
            mock_conn._bound_loop = asyncio.get_running_loop()
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
        mock_conn._bound_loop = None
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
        mock_conn._bound_loop = None
        mock_conn._check_in_use = MagicMock()

        async def failing_execute(sql, params=None):
            if "ROLLBACK" in sql:
                raise Exception("connection lost")
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

    async def test_cancelled_error_during_reset_does_not_leak(self) -> None:
        """CancelledError during ROLLBACK in _reset_connection must not leak connections."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        real_conn = DqliteConnection("127.0.0.1:9001")
        real_conn._protocol = MagicMock()
        real_conn._db_id = 1
        real_conn._bound_loop = asyncio.get_running_loop()
        real_conn._in_transaction = False
        real_conn._protocol.close = MagicMock()
        real_conn._protocol.wait_closed = AsyncMock()

        # execute succeeds normally, but ROLLBACK raises CancelledError
        async def exec_or_cancel(db_id, sql, params=None):
            if "ROLLBACK" in str(sql):
                raise asyncio.CancelledError()
            return (0, 0)

        real_conn._protocol.exec_sql = exec_or_cancel

        with patch.object(pool._cluster, "connect", return_value=real_conn):
            await pool.initialize()

        initial_size = pool._size

        # Use the connection with a transaction open, then raise to trigger cleanup
        try:
            async with pool.acquire() as conn:
                conn._in_transaction = True
                conn._tx_owner = asyncio.current_task()
                raise ValueError("app error triggers pool cleanup with ROLLBACK")
        except (ValueError, asyncio.CancelledError):
            pass

        # After cleanup: connection should be destroyed and size decremented
        assert pool._size == initial_size - 1, (
            f"_size should have decremented from {initial_size} to {initial_size - 1}, "
            f"got {pool._size}. CancelledError during ROLLBACK leaked the connection."
        )
        assert real_conn._pool_released, (
            "_pool_released should be True after connection cleanup, "
            "but CancelledError during ROLLBACK skipped setting it."
        )

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
        conn._bound_loop = asyncio.get_running_loop()
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
        mock_conn._bound_loop = None
        mock_conn._pool_released = False
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

    async def test_reset_connection_returns_false_on_cancelled_error(self) -> None:
        """_reset_connection must return False (not raise) when ROLLBACK is cancelled."""
        import asyncio

        from dqliteclient.connection import DqliteConnection

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        conn = DqliteConnection("127.0.0.1:9001")
        conn._protocol = MagicMock()
        conn._db_id = 1
        conn._bound_loop = asyncio.get_running_loop()
        conn._in_transaction = True

        async def cancel_on_rollback(db_id, sql, params=None):
            if "ROLLBACK" in str(sql):
                raise asyncio.CancelledError()
            return (0, 0)

        conn._protocol.exec_sql = cancel_on_rollback
        conn._protocol.close = MagicMock()
        conn._protocol.wait_closed = AsyncMock()

        # _reset_connection should catch CancelledError and return False,
        # not let it propagate
        result = await pool._reset_connection(conn)
        assert result is False, (
            f"_reset_connection should return False when ROLLBACK raises "
            f"CancelledError, but got {result}. CancelledError escaped the handler."
        )

    async def test_escaped_reference_rejected_after_release(self) -> None:
        """Using a connection after it's returned to the pool must raise InterfaceError."""
        import asyncio

        from dqliteclient.connection import DqliteConnection
        from dqliteclient.exceptions import InterfaceError

        pool = ConnectionPool(["localhost:9001"], max_size=1)

        real_conn = DqliteConnection("127.0.0.1:9001")
        real_conn._protocol = MagicMock()
        real_conn._db_id = 1
        real_conn._bound_loop = asyncio.get_running_loop()
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
        real_conn._bound_loop = asyncio.get_running_loop()
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
        real_conn._bound_loop = asyncio.get_running_loop()
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
        real_conn._bound_loop = asyncio.get_running_loop()
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
        conn._bound_loop = asyncio.get_running_loop()
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
