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
