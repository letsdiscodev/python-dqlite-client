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

    async def test_handshake_generates_unique_client_id_when_unspecified(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
        welcome_response: bytes,
    ) -> None:
        """When no client_id is passed, handshake must generate a unique non-zero id.

        Every connection defaulting to client_id=0 collapses all clients in
        the server's per-client metrics and tracing. The client must opaquely
        generate a random id per connection.
        """
        mock_reader.read.return_value = welcome_response

        p1 = DqliteProtocol(mock_reader, mock_writer)
        await p1.handshake()
        p2 = DqliteProtocol(mock_reader, mock_writer)
        await p2.handshake()

        assert p1._client_id != 0
        assert p2._client_id != 0
        assert p1._client_id != p2._client_id

    async def test_handshake_caps_heartbeat_timeout(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
    ) -> None:
        """A huge heartbeat_timeout should be capped to prevent timeout bypass."""
        from dqlitewire.messages import WelcomeResponse

        # Server sends an absurdly large heartbeat timeout (e.g., corrupted value)
        huge_timeout_ms = 10_000_000  # 10000 seconds
        mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=huge_timeout_ms).encode()

        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=10.0)
        await protocol.handshake()

        # The read timeout should be capped, not set to 10000 seconds
        assert protocol._timeout <= 300.0

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

    async def test_finalize_wrong_response_type(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """finalize() should reject non-EmptyResponse (catches protocol desync)."""
        from dqlitewire.messages import DbResponse

        # Server sends wrong response type
        mock_reader.read.return_value = DbResponse(db_id=99).encode()

        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.finalize(1, 1)

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

    async def test_exec_sql_with_empty_tuple_params(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
        result_response: bytes,
    ) -> None:
        """Empty tuple params should be preserved, not replaced with []."""
        mock_reader.read.return_value = result_response

        # This should not raise -- empty tuple is a valid Sequence
        await protocol.exec_sql(1, "SELECT 1", params=())

        # Verify the request was written (meaning params=() was accepted)
        mock_writer.write.assert_called()

    async def test_exec_sql_reads_single_response(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """exec_sql reads exactly one ResultResponse — it must not drain extras.

        The C dqlite server emits exactly one RESULT per EXEC_SQL call, even
        for multi-statement SQL: handle_exec_sql_done_cb recurses internally
        on exec->tail and calls SUCCESS(..., RESULT) only after the last
        statement completes. fill_result reports sqlite3_changes() of the
        last statement only.

        If any additional response bytes are buffered (server protocol
        violation, pipelining, test artefact), the client must leave them
        alone rather than silently consuming them.
        """
        from dqlitewire.messages import ResultResponse

        result1 = ResultResponse(last_insert_id=1, rows_affected=1)
        stray = ResultResponse(last_insert_id=99, rows_affected=99)
        mock_reader.read.return_value = result1.encode() + stray.encode()

        last_id, rows_affected = await protocol.exec_sql(1, "INSERT INTO t VALUES (1)")

        assert last_id == 1
        assert rows_affected == 1
        # The stray message must still be buffered — not silently consumed.
        assert protocol._decoder.has_message()

    async def test_query_sql_reads_single_response(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """query_sql reads exactly one RowsResponse — it must not drain extras.

        The C server rejects multi-statement SELECT with a FailureResponse
        ("nonempty statement tail") rather than sending multiple RowsResponses.
        Any stray post-response bytes must be left alone, not silently
        consumed.
        """
        from dqlitewire.constants import ValueType
        from dqlitewire.messages import RowsResponse

        rows1 = RowsResponse(
            column_names=["a"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            has_more=False,
        )
        stray = RowsResponse(
            column_names=["b"],
            column_types=[ValueType.INTEGER],
            rows=[[2]],
            has_more=False,
        )
        mock_reader.read.return_value = rows1.encode() + stray.encode()

        columns, rows = await protocol.query_sql(1, "SELECT 1")

        assert columns == ["a"]
        assert rows == [[1]]
        # Stray response must remain in the buffer, not silently consumed.
        assert protocol._decoder.has_message()

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

    async def test_query_sql_multipart(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Multi-part ROWS: initial frame + continuation drains correctly."""
        from dqlitewire.constants import (
            ROW_DONE_MARKER,
            ROW_PART_MARKER,
            ValueType,
        )
        from dqlitewire.messages.base import Header
        from dqlitewire.tuples import encode_row_header, encode_row_values
        from dqlitewire.types import encode_text, encode_uint64

        types = [ValueType.INTEGER, ValueType.TEXT]

        # Initial frame (has_more=True)
        body1 = encode_uint64(2)
        body1 += encode_text("id") + encode_text("name")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1, "alice"], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        h1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)

        # Continuation frame (has_more=False) — C server always
        # includes column_count + column_names in every ROWS frame
        body2 = encode_uint64(2)
        body2 += encode_text("id") + encode_text("name")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2, "bob"], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        h2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)

        # Feed both frames in one chunk
        all_bytes = h1.encode() + body1 + h2.encode() + body2
        mock_reader.read.return_value = all_bytes

        columns, rows = await protocol.query_sql(1, "SELECT id, name FROM t")

        assert columns == ["id", "name"]
        assert len(rows) == 2
        assert rows[0] == [1, "alice"]
        assert rows[1] == [2, "bob"]

    async def test_get_leader(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        leader_response: bytes,
    ) -> None:
        mock_reader.read.return_value = leader_response

        node_id, address = await protocol.get_leader()

        assert node_id == 1
        assert address == "localhost:9001"

    async def test_prepare(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        from dqlitewire.messages import StmtResponse

        mock_reader.read.return_value = StmtResponse(db_id=1, stmt_id=1, num_params=2).encode()

        stmt_id, num_params = await protocol.prepare(1, "INSERT INTO t VALUES (?, ?)")

        assert stmt_id == 1
        assert num_params == 2

    async def test_finalize(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        from dqlitewire.messages import EmptyResponse

        mock_reader.read.return_value = EmptyResponse().encode()

        # Should not raise
        await protocol.finalize(1, 1)

    async def test_connection_closed_during_read(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        mock_reader.read.return_value = b""

        from dqliteclient.exceptions import DqliteConnectionError

        with pytest.raises(DqliteConnectionError, match="Connection closed"):
            await protocol.exec_sql(1, "SELECT 1")

    async def test_read_timeout(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
    ) -> None:
        """Protocol reads should time out instead of blocking forever."""
        import asyncio

        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=0.1)

        # Simulate a server that hangs (never returns data)
        async def hang_forever(*args, **kwargs):
            await asyncio.sleep(100)
            return b""

        mock_reader.read.side_effect = hang_forever

        from dqliteclient.exceptions import DqliteConnectionError

        with pytest.raises(DqliteConnectionError, match="timed out"):
            await protocol.exec_sql(1, "SELECT 1")

    async def test_close(
        self,
        protocol: DqliteProtocol,
        mock_writer: MagicMock,
    ) -> None:
        protocol.close()
        mock_writer.close.assert_called_once()

        await protocol.wait_closed()
        mock_writer.wait_closed.assert_called_once()
