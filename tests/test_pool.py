"""Tests for connection pooling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class TestConnectionPool:
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

        async def hold_connection():
            async with pool.acquire() as conn:
                await asyncio.sleep(10)  # Hold forever

        task = asyncio.create_task(hold_connection())
        await asyncio.sleep(0.01)  # Let task acquire

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Connection should have been closed, size decremented
        mock_conn.close.assert_called()
        assert pool._size < initial_size


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
