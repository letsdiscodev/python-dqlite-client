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

    async def test_cancellation_does_not_leak_connection(self) -> None:
        """Cancelling a task that holds a connection should clean it up."""
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
                await asyncio.sleep(10)  # Hold forever

        task = asyncio.create_task(hold_connection())
        await acquired.wait()  # Deterministic: wait until connection is acquired

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Connection should have been closed, size decremented
        mock_conn.close.assert_called()
        assert pool._size < initial_size

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

    async def test_close_handles_checked_out_connections(self) -> None:
        """close() should close in-flight connections, not just idle ones."""
        pool = ConnectionPool(["localhost:9001"], max_size=2)

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
            await pool.initialize()  # Creates mock_conn1

        # Acquire a connection (checks it out)
        ctx = pool.acquire()
        await ctx.__aenter__()

        # Close the pool while connection is checked out
        await pool.close()

        # The checked-out connection should have been closed
        mock_conn1.close.assert_called()


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
