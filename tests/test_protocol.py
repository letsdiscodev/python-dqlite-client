"""Tests for low-level protocol handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import (
    FailureResponse,
)


class TestDqliteProtocol:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    @pytest.mark.parametrize(
        "err",
        [
            ConnectionError("bare connection error"),
            ConnectionResetError("peer reset"),
            BrokenPipeError("pipe gone"),
            OSError("generic os error"),
            RuntimeError("Transport is closed"),
        ],
    )
    async def test_writer_drain_errors_are_wrapped(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
        err: BaseException,
    ) -> None:
        """writer.drain() transport errors surface as DqliteConnectionError
        with the original attached via __cause__."""
        from dqliteclient.exceptions import DqliteConnectionError

        mock_writer.drain = AsyncMock(side_effect=err)

        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol.handshake()
        assert exc_info.value.__cause__ is err

    async def test_writer_drain_unrelated_exception_propagates_unwrapped(
        self,
        protocol: DqliteProtocol,
        mock_writer: MagicMock,
    ) -> None:
        """Exceptions outside the transport-error tuple propagate unchanged,
        so an over-collapse to catching Exception would fail this pin."""
        err = ValueError("unrelated")
        mock_writer.drain = AsyncMock(side_effect=err)

        with pytest.raises(ValueError) as exc_info:
            await protocol.handshake()
        assert exc_info.value is err

    @pytest.mark.parametrize(
        "err",
        [
            ConnectionError("bare connection error mid-read"),
            ConnectionResetError("peer reset mid-read"),
            BrokenPipeError("pipe gone mid-read"),
            OSError("generic os error mid-read"),
            RuntimeError("Transport is closed mid-read"),
        ],
    )
    async def test_reader_errors_are_wrapped(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        err: BaseException,
    ) -> None:
        """_read_data transport errors also surface as DqliteConnectionError."""
        from dqliteclient.exceptions import DqliteConnectionError

        mock_reader.read.side_effect = err
        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol._read_data()
        assert exc_info.value.__cause__ is err

    async def test_reader_unrelated_exception_propagates_unwrapped(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Exceptions outside the transport-error tuple propagate unchanged."""
        err = ValueError("unrelated")
        mock_reader.read.side_effect = err

        with pytest.raises(ValueError) as exc_info:
            await protocol._read_data()
        assert exc_info.value is err

    async def test_read_timeout_preserves_cause(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """A per-read timeout surfaces as DqliteConnectionError with __cause__
        set to the underlying TimeoutError."""
        import asyncio

        from dqliteclient.exceptions import DqliteConnectionError

        protocol._timeout = 0.05

        async def hang(_n: int) -> bytes:
            await asyncio.sleep(10)
            return b""

        mock_reader.read.side_effect = hang
        with pytest.raises(DqliteConnectionError, match="Server read timed out") as exc_info:
            await protocol._read_data()
        assert isinstance(exc_info.value.__cause__, TimeoutError)

    async def test_read_response_enforces_operation_deadline(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Even if each read returns under the per-read timeout, the cumulative
        operation deadline must fire."""
        import asyncio
        import time

        # Set both timeouts so the deadline (keyed off _read_timeout) reflects
        # the tightened budget.
        protocol._timeout = 0.2
        protocol._read_timeout = 0.2

        async def drip_forever(_n: int) -> bytes:
            await asyncio.sleep(0.1)
            return b"\x00"  # never completes a message

        mock_reader.read.side_effect = drip_forever

        t0 = time.monotonic()
        with pytest.raises(DqliteConnectionError, match="deadline|timed out"):
            await protocol._read_response()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, (
            f"_read_response must bail at the operation deadline; took {elapsed:.3f}s"
        )

    async def test_read_timeout_widening_actually_extends_operation_deadline(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Setting ``_read_timeout`` genuinely extends the per-operation read
        deadline (the widening is not dead code): a read longer than ``_timeout``
        but under ``_read_timeout`` completes cleanly."""
        import asyncio

        from dqlitewire.messages import EmptyResponse

        protocol._timeout = 0.05  # tight write-side
        protocol._read_timeout = 5.0  # widened read-side
        body = EmptyResponse().encode()

        sent = [False]

        async def slow_first_read(_n: int) -> bytes:
            # Longer than _timeout but within _read_timeout; deliver full response.
            if not sent[0]:
                sent[0] = True
                await asyncio.sleep(0.2)
                return body
            return b""

        mock_reader.read.side_effect = slow_first_read

        # Must NOT raise: pre-fix this raised TimeoutError at 0.05s.
        await protocol._read_response()

    async def test_read_data_deadline_already_in_past_raises_immediately(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """``_read_data`` short-circuits an already-passed deadline before
        ``wait_for`` to dodge ``wait_for(timeout=0)`` drift and negative-timeout
        ValueError on some builds; must raise without entering the reader."""
        import asyncio

        loop = asyncio.get_running_loop()
        past_deadline = loop.time() - 1.0

        with pytest.raises(DqliteConnectionError, match="exceeded.*deadline"):
            await protocol._read_data(deadline=past_deadline)

        assert not mock_reader.read.called, (
            "_read_data must not enter the reader when deadline is already past"
        )

    async def test_read_data_deadline_in_future_still_uses_wait_for(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """A future deadline must NOT take the fast-path; the reader is entered."""
        import asyncio

        loop = asyncio.get_running_loop()
        future_deadline = loop.time() + 5.0

        mock_reader.read.return_value = b"\x00" * 16
        data = await protocol._read_data(deadline=future_deadline)
        assert data == b"\x00" * 16
        assert mock_reader.read.called

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
        """Without a client_id, handshake generates a unique non-zero id;
        defaulting to 0 would collapse all clients in server-side metrics/tracing."""
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
        """A huge heartbeat_timeout is capped to prevent timeout bypass."""
        from dqlitewire.messages import WelcomeResponse

        huge_timeout_ms = 10_000_000  # 10000 seconds
        mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=huge_timeout_ms).encode()

        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=10.0)
        await protocol.handshake()

        assert protocol._timeout <= 300.0

    async def test_handshake_emits_debug_when_trust_widens_timeout(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
        caplog,
    ) -> None:
        """``trust_server_heartbeat=True`` that widens the deadline emits a DEBUG
        log so an operator can confirm the opt-in took effect."""
        import logging as _logging

        from dqlitewire.messages import WelcomeResponse

        mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=30_000).encode()
        protocol = DqliteProtocol(
            mock_reader, mock_writer, timeout=5.0, trust_server_heartbeat=True
        )
        caplog.set_level(_logging.DEBUG, logger="dqliteclient.protocol")
        await protocol.handshake()

        messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.protocol"]
        assert any("widened per-read timeout" in m for m in messages)

    async def test_handshake_silent_when_trust_disabled(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
        caplog,
    ) -> None:
        """Default (``trust_server_heartbeat=False``) must not log for the
        server-advertised heartbeat, even when the value is useful."""
        import logging as _logging

        from dqlitewire.messages import WelcomeResponse

        mock_reader.read.return_value = WelcomeResponse(heartbeat_timeout=30_000).encode()
        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=5.0)
        caplog.set_level(_logging.DEBUG, logger="dqliteclient.protocol")
        await protocol.handshake()

        messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.protocol"]
        assert not any("widened" in m for m in messages)

    async def test_handshake_failure(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        failure = FailureResponse(code=1, message="auth failed").encode()
        mock_reader.read.return_value = failure

        with pytest.raises(OperationalError, match="Handshake failed") as ei:
            await protocol.handshake()
        assert ei.value.code == 1

    async def test_handshake_failure_includes_code_and_address(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
    ) -> None:
        """Handshake FailureResponse surfaces the server code and peer address
        so operators can distinguish e.g. DQLITE_PARSE from DQLITE_NOTLEADER."""
        failure = FailureResponse(code=101, message="bad protocol version").encode()
        mock_reader.read.return_value = failure
        protocol = DqliteProtocol(mock_reader, mock_writer, address="leader.example:9001")

        with pytest.raises(OperationalError) as ei:
            await protocol.handshake()

        message = str(ei.value)
        assert "[101]" in message
        assert "bad protocol version" in message
        assert "leader.example:9001" in message
        assert ei.value.code == 101

    async def test_handshake_failure_empty_message_uses_placeholder(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """An empty server message surfaces the "(no diagnostic from server)"
        placeholder rather than a bare "[1] " rendering."""
        failure = FailureResponse(code=1, message="").encode()
        mock_reader.read.return_value = failure

        with pytest.raises(OperationalError) as ei:
            await protocol.handshake()

        assert "(no diagnostic from server)" in str(ei.value)

    async def test_open_database(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
        db_response: bytes,
    ) -> None:
        mock_reader.read.return_value = db_response

        db_id = await protocol.open_database("test.db")

        assert db_id == 0

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
        """finalize() rejects non-EmptyResponse (catches protocol desync)."""
        from dqlitewire.messages import DbResponse

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
        """Empty tuple params are preserved, not replaced with []."""
        mock_reader.read.return_value = result_response

        await protocol.exec_sql(1, "SELECT 1", params=())

        mock_writer.write.assert_called()

    async def test_exec_sql_rejects_extra_response(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """exec_sql reads exactly one ResultResponse (the C server emits one
        RESULT per EXEC_SQL even for multi-statement SQL); extra bytes must
        invalidate rather than be buffered for the next RPC to misconsume."""
        from dqliteclient.exceptions import ProtocolError
        from dqlitewire.messages import ResultResponse

        result1 = ResultResponse(last_insert_id=1, rows_affected=1)
        stray = ResultResponse(last_insert_id=99, rows_affected=99)
        mock_reader.read.return_value = result1.encode() + stray.encode()

        with pytest.raises(ProtocolError, match="extra response"):
            await protocol.exec_sql(1, "INSERT INTO t VALUES (1)")

    async def test_query_sql_reads_single_response(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """query_sql reads exactly one RowsResponse and must not drain extras
        (the C server rejects multi-statement SELECT rather than sending more)."""
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
        # Stray response must remain buffered, not silently consumed.
        assert protocol._decoder.has_message()

    async def test_query_sql_raises_if_continuation_has_no_progress(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """A ROWS continuation with has_more=True but 0 rows must raise rather
        than loop forever (pathological case: column header alone exceeds the
        page buffer, so query__batch sends a partial frame with no rows)."""
        from dqlitewire.constants import ValueType
        from dqlitewire.messages import RowsResponse

        first = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[],
            has_more=True,
        )
        stuck = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[],
            has_more=True,
        )
        mock_reader.read.return_value = first.encode() + stuck.encode()

        with pytest.raises(ProtocolError, match="no progress|no rows"):
            await protocol.query_sql(1, "SELECT x FROM wide_table")

    async def test_query_sql_typed_returns_column_types(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """query_sql_typed returns wire ValueType ints (column + per-row)
        alongside names+rows for DBAPI cursor.description and per-row dispatch."""
        from dqlitewire.constants import ValueType
        from dqlitewire.messages import RowsResponse

        response = RowsResponse(
            column_names=["a", "b"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            row_types=[[ValueType.INTEGER, ValueType.TEXT]],
            rows=[[1, "x"]],
            has_more=False,
        )
        mock_reader.read.return_value = response.encode()

        names, types, row_types, rows = await protocol.query_sql_typed(1, "SELECT a, b FROM t")
        assert names == ["a", "b"]
        assert types == [int(ValueType.INTEGER), int(ValueType.TEXT)]
        assert row_types == [[int(ValueType.INTEGER), int(ValueType.TEXT)]]
        assert rows == [[1, "x"]]

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

        # Initial frame (has_more=True).
        body1 = encode_uint64(2)
        body1 += encode_text("id") + encode_text("name")
        body1 += encode_row_header(types)
        body1 += encode_row_values([1, "alice"], types)
        body1 += encode_uint64(ROW_PART_MARKER)
        h1 = Header(size_words=len(body1) // 8, msg_type=7, schema=0)

        # Continuation frame (has_more=False); C server repeats column
        # count + names in every ROWS frame.
        body2 = encode_uint64(2)
        body2 += encode_text("id") + encode_text("name")
        body2 += encode_row_header(types)
        body2 += encode_row_values([2, "bob"], types)
        body2 += encode_uint64(ROW_DONE_MARKER)
        h2 = Header(size_words=len(body2) // 8, msg_type=7, schema=0)

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

    async def test_prepare_db_id_mismatch_raises_protocol_error_with_wire_decode_prefix(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """A StmtResponse db_id mismatch (registry drift) raises ProtocolError
        prefixed with "wire decode failed:" so SA's is_disconnect matcher routes
        it through pool-invalidate instead of keeping the broken slot."""
        from dqlitewire.messages import StmtResponse

        # Prepare against db_id=1 but server returns db_id=99 (drift signal).
        mock_reader.read.return_value = StmtResponse(db_id=99, stmt_id=42, num_params=0).encode()

        with pytest.raises(ProtocolError) as exc_info:
            await protocol.prepare(1, "SELECT 1")

        message = str(exc_info.value)
        assert message.startswith("wire decode failed:"), (
            f"prefix is load-bearing for SA's is_disconnect matcher; got: {message!r}"
        )
        assert "db_id 99" in message
        assert "db_id 1" in message

    async def test_finalize(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        from dqlitewire.messages import EmptyResponse

        mock_reader.read.return_value = EmptyResponse().encode()

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
        """Protocol reads time out instead of blocking forever."""
        import asyncio

        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=0.1)

        async def hang_forever(*args, **kwargs):
            await asyncio.sleep(100)
            return b""

        mock_reader.read.side_effect = hang_forever

        from dqliteclient.exceptions import DqliteConnectionError

        with pytest.raises(DqliteConnectionError, match="timed out"):
            await protocol.exec_sql(1, "SELECT 1")

    async def test_timeout_message_includes_peer_address(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
    ) -> None:
        """When ``address`` is set, transport errors embed it so an operator
        can tell which node hung."""
        import asyncio

        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=0.05, address="node-a:9001")

        async def hang_forever(*args: object, **kwargs: object) -> bytes:
            await asyncio.sleep(100)
            return b""

        mock_reader.read.side_effect = hang_forever

        from dqliteclient.exceptions import DqliteConnectionError

        with pytest.raises(DqliteConnectionError, match=r"to node-a:9001"):
            await protocol.exec_sql(1, "SELECT 1")

    async def test_timeout_message_unchanged_when_address_absent(
        self,
        mock_reader: AsyncMock,
        mock_writer: MagicMock,
    ) -> None:
        """Without an address, the message must not include a stray "to None"."""
        import asyncio

        protocol = DqliteProtocol(mock_reader, mock_writer, timeout=0.05)

        async def hang_forever(*args: object, **kwargs: object) -> bytes:
            await asyncio.sleep(100)
            return b""

        mock_reader.read.side_effect = hang_forever

        from dqliteclient.exceptions import DqliteConnectionError

        with pytest.raises(DqliteConnectionError) as exc_info:
            await protocol.exec_sql(1, "SELECT 1")
        assert " to " not in str(exc_info.value), (
            f"Address-less protocol should not inject 'to None'; got: {exc_info.value!s}"
        )

    async def test_close(
        self,
        protocol: DqliteProtocol,
        mock_writer: MagicMock,
    ) -> None:
        protocol.close()
        mock_writer.close.assert_called_once()

        await protocol.wait_closed()
        mock_writer.wait_closed.assert_called_once()
