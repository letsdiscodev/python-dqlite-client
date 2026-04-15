"""Tests for high-level connection interface."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


class TestParseAddress:
    def test_ipv4(self) -> None:
        from dqliteclient.connection import _parse_address

        assert _parse_address("localhost:9001") == ("localhost", 9001)

    def test_ipv4_ip(self) -> None:
        from dqliteclient.connection import _parse_address

        assert _parse_address("192.168.1.1:9001") == ("192.168.1.1", 9001)

    def test_ipv6_bracketed(self) -> None:
        from dqliteclient.connection import _parse_address

        assert _parse_address("[::1]:9001") == ("::1", 9001)

    def test_ipv6_full_bracketed(self) -> None:
        from dqliteclient.connection import _parse_address

        assert _parse_address("[2001:db8::1]:9001") == ("2001:db8::1", 9001)

    def test_bare_hostname_raises(self) -> None:
        from dqliteclient.connection import _parse_address

        with pytest.raises(ValueError, match="expected.*host:port"):
            _parse_address("localhost")

    def test_invalid_port_raises(self) -> None:
        from dqliteclient.connection import _parse_address

        with pytest.raises(ValueError, match="not a number"):
            _parse_address("localhost:abc")

    def test_ipv6_no_port_raises(self) -> None:
        from dqliteclient.connection import _parse_address

        with pytest.raises(ValueError, match="expected.*host.*port"):
            _parse_address("[::1]")


class TestDqliteConnection:
    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            DqliteConnection("localhost:9001", timeout=0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            DqliteConnection("localhost:9001", timeout=-1)

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
            pytest.raises(DqliteConnectionError, match="timed out"),
        ):
            await conn.connect()

    async def test_connect_refused(self) -> None:
        conn = DqliteConnection("localhost:9001")

        with (
            patch(
                "asyncio.open_connection",
                side_effect=OSError("Connection refused"),
            ),
            pytest.raises(DqliteConnectionError, match="Failed to connect"),
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

        with pytest.raises(DqliteConnectionError, match="Not connected"):
            await conn.execute("SELECT 1")

    async def test_nested_transaction_raises(self) -> None:
        """Nested transaction() should raise, not silently no-op."""
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import DbResponse, ResultResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
            ResultResponse(last_insert_id=0, rows_affected=0).encode(),  # BEGIN
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        # Mock execute for BEGIN
        async def mock_execute(sql: str, params=None):
            return (0, 0)

        conn.execute = mock_execute  # type: ignore[assignment]

        from dqliteclient.exceptions import OperationalError

        async with conn.transaction():
            with pytest.raises(OperationalError, match="[Nn]ested"):
                async with conn.transaction():
                    pass

    async def test_fetch_not_connected(self) -> None:
        conn = DqliteConnection("localhost:9001")

        with pytest.raises(DqliteConnectionError, match="Not connected"):
            await conn.fetch("SELECT 1")

    async def test_transaction_rollback_failure_preserves_original_exception(self) -> None:
        """If ROLLBACK fails, the original exception should still propagate."""
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import DbResponse, ResultResponse, WelcomeResponse

        # Handshake + open_database + BEGIN succeed
        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
            ResultResponse(last_insert_id=0, rows_affected=0).encode(),  # BEGIN
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        # Mock execute: BEGIN succeeds, ROLLBACK fails
        call_log: list[str] = []

        async def mock_execute(sql: str, params=None):
            call_log.append(sql)
            if "ROLLBACK" in sql:
                raise OSError("Connection lost")
            return (0, 0)

        conn.execute = mock_execute  # type: ignore[assignment]

        with pytest.raises(ValueError, match="user error"):
            async with conn.transaction():
                raise ValueError("user error")

        # ROLLBACK was attempted
        assert "ROLLBACK" in call_log
        # _in_transaction was cleaned up
        assert not conn._in_transaction

    async def test_cancellation_invalidates_connection(self) -> None:
        """CancelledError during a query must invalidate the connection."""
        import asyncio

        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        assert conn.is_connected

        # Make the reader hang forever (will be cancelled)
        read_entered = asyncio.Event()

        async def hanging_read(*args, **kwargs):
            read_entered.set()
            await asyncio.sleep(100)

        mock_reader.read.side_effect = hanging_read

        async def do_execute():
            await conn.execute("INSERT INTO t VALUES (1)")

        task = asyncio.create_task(do_execute())
        await read_entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Connection must be invalidated — the decoder may have partial data
        assert not conn.is_connected, (
            "Connection should be invalidated after CancelledError to prevent "
            "decoder corruption from partial reads"
        )

    async def test_connection_invalidated_after_protocol_error(self) -> None:
        """After a connection error, is_connected should return False."""
        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        assert conn.is_connected

        # Now make the reader return empty (connection closed)
        mock_reader.read.side_effect = [b""]

        with pytest.raises(DqliteConnectionError):
            await conn.execute("SELECT 1")

        # Connection should be invalidated
        assert not conn.is_connected

    async def test_invalidate_closes_transport(self) -> None:
        """_invalidate() should close the underlying transport to avoid socket leaks."""
        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        # Trigger invalidation via a connection error
        mock_reader.read.side_effect = [b""]

        with pytest.raises(DqliteConnectionError):
            await conn.execute("SELECT 1")

        # The writer should have been closed to release the socket
        mock_writer.close.assert_called()

    async def test_fetchone_returns_first_row(self) -> None:
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.constants import ValueType
        from dqlitewire.messages import DbResponse, RowsResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
            RowsResponse(
                column_names=["id", "name"],
                column_types=[ValueType.INTEGER, ValueType.TEXT],
                rows=[[1, "first"], [2, "second"]],
                has_more=False,
            ).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        mock_reader.read.side_effect = [responses[2]]
        result = await conn.fetchone("SELECT * FROM t")
        assert result == {"id": 1, "name": "first"}

    async def test_fetchone_returns_none_for_empty(self) -> None:
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.constants import ValueType
        from dqlitewire.messages import DbResponse, RowsResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        empty_response = RowsResponse(
            column_names=["id"],
            column_types=[ValueType.INTEGER],
            rows=[],
            has_more=False,
        ).encode()
        mock_reader.read.side_effect = [empty_response]
        result = await conn.fetchone("SELECT * FROM t WHERE 1=0")
        assert result is None

    async def test_fetchall_returns_lists(self) -> None:
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.constants import ValueType
        from dqlitewire.messages import DbResponse, RowsResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        rows_response = RowsResponse(
            column_names=["id", "name"],
            column_types=[ValueType.INTEGER, ValueType.TEXT],
            rows=[[1, "a"], [2, "b"]],
            has_more=False,
        ).encode()
        mock_reader.read.side_effect = [rows_response]
        result = await conn.fetchall("SELECT * FROM t")
        assert result == [[1, "a"], [2, "b"]]

    async def test_fetchval_returns_first_column(self) -> None:
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.constants import ValueType
        from dqlitewire.messages import DbResponse, RowsResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        rows_response = RowsResponse(
            column_names=["count"],
            column_types=[ValueType.INTEGER],
            rows=[[42]],
            has_more=False,
        ).encode()
        mock_reader.read.side_effect = [rows_response]
        result = await conn.fetchval("SELECT count(*) FROM t")
        assert result == 42

    async def test_fetchval_returns_none_for_empty(self) -> None:
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.constants import ValueType
        from dqlitewire.messages import DbResponse, RowsResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        empty_response = RowsResponse(
            column_names=["id"],
            column_types=[ValueType.INTEGER],
            rows=[],
            has_more=False,
        ).encode()
        mock_reader.read.side_effect = [empty_response]
        result = await conn.fetchval("SELECT id FROM t WHERE 1=0")
        assert result is None

    async def test_transaction_rollback_on_cancellation(self) -> None:
        """CancelledError inside a transaction must trigger ROLLBACK."""
        import asyncio

        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        # Track which SQL statements are executed
        call_log: list[str] = []

        async def mock_execute(sql: str, params=None):
            call_log.append(sql)
            return (0, 0)

        conn.execute = mock_execute  # type: ignore[assignment]

        async def cancelled_transaction():
            async with conn.transaction():
                await asyncio.sleep(10)  # Will be cancelled here

        task = asyncio.create_task(cancelled_transaction())
        await asyncio.sleep(0)  # Let the task enter the transaction
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # ROLLBACK must have been issued
        assert "ROLLBACK" in call_log, (
            f"ROLLBACK was not issued on CancelledError. Calls: {call_log}"
        )
        # _in_transaction must be cleaned up
        assert not conn._in_transaction

    async def test_connect_cancellation_cleans_up_protocol(self) -> None:
        """Cancelling connect() during handshake must close the transport."""
        import asyncio

        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        handshake_entered = asyncio.Event()

        async def slow_handshake(*args, **kwargs):
            handshake_entered.set()
            await asyncio.sleep(10)  # Will be cancelled

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch("dqliteclient.connection.DqliteProtocol") as MockProto,
        ):
            proto_instance = MagicMock()
            proto_instance.handshake = AsyncMock(side_effect=slow_handshake)
            proto_instance.close = MagicMock()
            MockProto.return_value = proto_instance

            task = asyncio.create_task(conn.connect())
            await handshake_entered.wait()
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

        # The protocol must have been closed to avoid socket leak
        proto_instance.close.assert_called()
        # The connection must not appear as connected
        assert not conn.is_connected

    async def test_not_leader_error_invalidates_connection(self) -> None:
        """OperationalError with 'not leader' code should invalidate the connection."""
        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import DbResponse, FailureResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        assert conn.is_connected

        # Server responds with "not leader" error
        # SQLITE_IOERR_NOT_LEADER = SQLITE_IOERR | (40 << 8) = 10250
        not_leader = FailureResponse(code=10250, message="not leader").encode()
        mock_reader.read.side_effect = [not_leader]

        from dqliteclient.exceptions import OperationalError

        with pytest.raises(OperationalError, match="not leader"):
            await conn.execute("INSERT INTO t VALUES (1)")

        # Connection should be invalidated after a leader error
        assert not conn.is_connected

    async def test_concurrent_coroutines_raises_interface_error(self) -> None:
        """Two coroutines using the same connection must raise InterfaceError."""
        import asyncio

        from dqliteclient.exceptions import InterfaceError

        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        # Make execute hang so two coroutines overlap
        first_entered = asyncio.Event()

        async def slow_exec_sql(db_id, sql, params=None):
            first_entered.set()
            await asyncio.sleep(10)
            return (0, 1)

        conn._protocol.exec_sql = AsyncMock(side_effect=slow_exec_sql)  # type: ignore[union-attr]

        errors: list[Exception] = []

        async def first_execute():
            await conn.execute("INSERT INTO t VALUES (1)")

        async def second_execute():
            await first_entered.wait()
            try:
                await conn.execute("INSERT INTO t VALUES (2)")
            except InterfaceError as e:
                errors.append(e)

        task1 = asyncio.create_task(first_execute())
        task2 = asyncio.create_task(second_execute())

        await task2  # second should raise InterfaceError
        task1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task1

        assert len(errors) == 1
        assert "another operation is in progress" in str(errors[0])

    async def test_concurrent_connect_raises_interface_error(self) -> None:
        """Two coroutines connecting the same object must raise InterfaceError."""
        import asyncio

        from dqliteclient.exceptions import InterfaceError

        conn = DqliteConnection("localhost:9001")

        gate = asyncio.Event()

        async def slow_open(*args, **kwargs):
            await gate.wait()
            reader = AsyncMock()
            writer = MagicMock()
            writer.drain = AsyncMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        errors: list[Exception] = []

        async def first_connect():
            with (
                patch("asyncio.open_connection", side_effect=slow_open),
                patch("dqliteclient.connection.DqliteProtocol") as MockProto,
            ):
                proto = MagicMock()
                proto.handshake = AsyncMock()
                proto.open_database = AsyncMock(return_value=1)
                proto.close = MagicMock()
                MockProto.return_value = proto
                await conn.connect()

        async def second_connect():
            await asyncio.sleep(0)  # Let first_connect start
            try:
                await conn.connect()
            except InterfaceError as e:
                errors.append(e)

        task1 = asyncio.create_task(first_connect())
        task2 = asyncio.create_task(second_connect())

        await asyncio.sleep(0)  # Let both start
        gate.set()

        await asyncio.gather(task1, task2, return_exceptions=True)

        assert len(errors) == 1
        assert "another operation is in progress" in str(errors[0])

    async def test_close_while_in_use_raises_interface_error(self) -> None:
        """close() must raise InterfaceError if an operation is in progress."""
        import asyncio

        from dqliteclient.exceptions import InterfaceError

        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        # Make execute hang so close() runs while execute is in progress
        execute_entered = asyncio.Event()

        async def slow_exec_sql(db_id, sql, params=None):
            execute_entered.set()
            await asyncio.sleep(10)
            return (0, 1)

        conn._protocol.exec_sql = AsyncMock(side_effect=slow_exec_sql)  # type: ignore[union-attr]

        async def do_execute():
            await conn.execute("INSERT INTO t VALUES (1)")

        task = asyncio.create_task(do_execute())
        await execute_entered.wait()

        # close() should raise because execute is in progress
        with pytest.raises(InterfaceError, match="another operation is in progress"):
            await conn.close()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_concurrent_transaction_raises_operational_error(self) -> None:
        """Second concurrent transaction() must raise OperationalError, not InterfaceError."""
        import asyncio

        from dqliteclient.exceptions import InterfaceError, OperationalError

        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        # Mock execute to track calls and allow concurrent entry
        begin_entered = asyncio.Event()

        async def mock_execute(sql: str, params=None):
            if sql == "BEGIN":
                begin_entered.set()
                await asyncio.sleep(0)  # yield to let second coroutine enter
            return (0, 0)

        conn.execute = mock_execute  # type: ignore[assignment]

        errors: list[Exception] = []

        async def tx_a():
            async with conn.transaction():
                await asyncio.sleep(1)

        async def tx_b():
            await begin_entered.wait()
            try:
                async with conn.transaction():
                    pass
            except (OperationalError, InterfaceError) as e:
                errors.append(e)

        task_a = asyncio.create_task(tx_a())
        task_b = asyncio.create_task(tx_b())

        await task_b  # should raise OperationalError about nested transactions

        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a

        assert len(errors) == 1
        # Must get OperationalError about nested transactions, NOT InterfaceError
        assert isinstance(errors[0], OperationalError), (
            f"Expected OperationalError about nested transactions, "
            f"got {type(errors[0]).__name__}: {errors[0]}"
        )
        assert "Nested" in str(errors[0]) or "nested" in str(errors[0])

    async def test_cross_event_loop_raises_interface_error(self) -> None:
        """Using a connection from a different event loop must raise InterfaceError."""
        import asyncio
        import threading

        from dqliteclient.exceptions import InterfaceError

        conn = DqliteConnection("localhost:9001")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import DbResponse, ResultResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            DbResponse(db_id=1).encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await conn.connect()

        # Now try to use the connection from a different event loop in another thread
        error_from_thread: Exception | None = None

        def run_in_other_loop():
            nonlocal error_from_thread

            async def use_conn():
                # Provide a fresh response so the operation could succeed
                # if the guard doesn't catch it
                mock_reader.read.side_effect = [
                    ResultResponse(last_insert_id=0, rows_affected=0).encode(),
                ]
                await conn.execute("SELECT 1")

            try:
                asyncio.run(use_conn())
            except InterfaceError as e:
                error_from_thread = e
            except Exception as e:
                error_from_thread = e

        thread = threading.Thread(target=run_in_other_loop)
        thread.start()
        thread.join(timeout=5)

        assert error_from_thread is not None, "Expected InterfaceError from cross-loop access"
        assert isinstance(error_from_thread, InterfaceError), (
            f"Expected InterfaceError, got {type(error_from_thread).__name__}: {error_from_thread}"
        )
        assert "event loop" in str(error_from_thread).lower()

    async def test_other_task_rejected_during_transaction(self) -> None:
        """Another task calling execute() during an active transaction must be rejected."""
        import asyncio

        from dqliteclient.exceptions import InterfaceError

        conn = DqliteConnection("localhost:9001")

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
            await conn.connect()

        # Mock execute for the transaction owner (task A) — needs to work
        a_inside_tx = asyncio.Event()

        async def mock_execute_a(sql: str, params=None):
            if sql not in ("BEGIN", "COMMIT", "ROLLBACK"):
                a_inside_tx.set()
                await asyncio.sleep(0)  # yield to let task B run
            return (0, 0)

        conn.execute = mock_execute_a  # type: ignore[assignment]

        errors: list[Exception] = []

        async def task_a():
            async with conn.transaction():
                await conn.execute("INSERT INTO t VALUES (1)")
                await asyncio.sleep(0.1)

        async def task_b():
            await a_inside_tx.wait()
            try:
                # Use _check_in_use directly — this is what real execute() calls
                conn._check_in_use()
            except InterfaceError as e:
                errors.append(e)

        t_a = asyncio.create_task(task_a())
        t_b = asyncio.create_task(task_b())

        await asyncio.gather(t_a, t_b, return_exceptions=True)

        assert len(errors) == 1, (
            "Task B should have been rejected when trying to use a connection "
            "that is in a transaction owned by task A"
        )
        assert isinstance(errors[0], InterfaceError)
        assert "transaction" in str(errors[0]).lower()
