"""Tests for high-level connection interface."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import ConnectionError


class TestDqliteConnection:
    def test_init(self) -> None:
        conn = DqliteConnection("localhost:9001", database="test", timeout=5.0)
        assert conn.address == "localhost:9001"
        assert not conn.is_connected

    async def test_connect_success(self) -> None:
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        # Mock the welcome and db responses
        from dqlitewire.messages import DbResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        assert conn.is_connected

        await conn.close()
        assert not conn.is_connected

    async def test_connect_timeout(self) -> None:
        import asyncio

        conn = DqliteConnection("localhost:9001", timeout=0.01)

        async def slow_connect(*args, **kwargs):
            await asyncio.sleep(1)
            return MagicMock(), MagicMock()

        with (
            patch("asyncio.open_connection", side_effect=slow_connect),
            pytest.raises(ConnectionError, match="timed out"),
        ):
            await conn.connect()

    async def test_connect_refused(self) -> None:
        conn = DqliteConnection("localhost:9001")

        with (
            patch(
                "asyncio.open_connection",
                side_effect=OSError("Connection refused"),
            ),
            pytest.raises(ConnectionError, match="Failed to connect"),
        ):
            await conn.connect()

    async def test_context_manager(self) -> None:
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import DbResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            async with DqliteConnection("localhost:9001") as conn:
                assert conn.is_connected

        assert not conn.is_connected

    async def test_execute_not_connected(self) -> None:
        conn = DqliteConnection("localhost:9001")

        with pytest.raises(ConnectionError, match="Not connected"):
            await conn.execute("SELECT 1")

    async def test_fetch_not_connected(self) -> None:
        conn = DqliteConnection("localhost:9001")

        with pytest.raises(ConnectionError, match="Not connected"):
            await conn.fetch("SELECT 1")
