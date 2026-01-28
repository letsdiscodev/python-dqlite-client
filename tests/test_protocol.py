"""Tests for low-level protocol handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import (
    FailureResponse,
)


class TestDqliteProtocol:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_handshake_success(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        welcome_response: bytes,
    ) -> None:
        mock_reader.read.return_value = welcome_response

        timeout = await protocol.handshake(client_id=42)

        assert timeout == 15000

    async def test_handshake_failure(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        failure = FailureResponse(code=1, message="auth failed").encode()
        mock_reader.read.return_value = failure

        with pytest.raises(ProtocolError, match="Handshake failed"):
            await protocol.handshake()

    async def test_open_database(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        db_response: bytes,
    ) -> None:
        mock_reader.read.return_value = db_response

        db_id = await protocol.open_database("test.db")

        assert db_id == 1

    async def test_open_database_failure(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        failure = FailureResponse(code=1, message="cannot open").encode()
        mock_reader.read.return_value = failure

        with pytest.raises(OperationalError) as exc_info:
            await protocol.open_database("test.db")

        assert exc_info.value.code == 1
        assert "cannot open" in exc_info.value.message

    async def test_exec_sql(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        result_response: bytes,
    ) -> None:
        mock_reader.read.return_value = result_response

        last_id, rows_affected = await protocol.exec_sql(1, "INSERT INTO t VALUES (1)")

        assert last_id == 1
        assert rows_affected == 1

    async def test_query_sql(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        rows_response: bytes,
    ) -> None:
        mock_reader.read.return_value = rows_response

        columns, rows = await protocol.query_sql(1, "SELECT * FROM t")

        assert columns == ["id", "name"]
        assert len(rows) == 1
        assert rows[0] == [1, "test"]

    async def test_close(
        self,
        protocol: DqliteProtocol,
        mock_writer: MagicMock,
    ) -> None:
        protocol.close()
        mock_writer.close.assert_called_once()

        await protocol.wait_closed()
        mock_writer.wait_closed.assert_called_once()
