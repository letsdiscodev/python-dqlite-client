"""Low-level protocol handler for dqlite."""

import asyncio
import secrets
from collections.abc import Sequence
from typing import Any

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqlitewire import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import ProtocolError as _WireProtocolError
from dqlitewire.messages import (
    ClientRequest,
    DbResponse,
    EmptyResponse,
    ExecSqlRequest,
    FailureResponse,
    FinalizeRequest,
    LeaderRequest,
    LeaderResponse,
    OpenRequest,
    PrepareRequest,
    QuerySqlRequest,
    ResultResponse,
    RowsResponse,
    StmtResponse,
    WelcomeResponse,
)
from dqlitewire.messages.base import Message


class DqliteProtocol:
    """Low-level protocol handler for a single dqlite connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        timeout: float = 15.0,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._decoder = MessageDecoder(is_request=False)
        self._client_id = 0
        self._heartbeat_timeout = 0
        self._timeout = timeout

    async def handshake(self, client_id: int | None = None) -> int:
        """Perform protocol handshake.

        If ``client_id`` is not provided, a random non-zero 63-bit id is
        generated so each connection is distinguishable in server logs,
        traces, and per-client metrics. Returns the heartbeat timeout
        from the server.
        """
        if client_id is None:
            client_id = secrets.randbits(63) or 1
        # Send protocol version + client registration together
        request = ClientRequest(client_id=client_id)
        self._writer.write(MessageEncoder().encode_handshake() + request.encode())
        await self._send()

        # Read welcome response
        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise ProtocolError(f"Handshake failed: {response.message}")

        if not isinstance(response, WelcomeResponse):
            raise ProtocolError(f"Expected WelcomeResponse, got {type(response).__name__}")

        self._client_id = client_id
        self._heartbeat_timeout = response.heartbeat_timeout
        # Use heartbeat timeout for subsequent reads if larger than default
        if response.heartbeat_timeout > 0:
            heartbeat_seconds = response.heartbeat_timeout / 1000.0
            # Cap to prevent a malicious/buggy server from disabling timeouts
            self._timeout = max(self._timeout, min(heartbeat_seconds, 300.0))
        return response.heartbeat_timeout

    async def get_leader(self) -> tuple[int, str]:
        """Request leader information.

        Returns (node_id, address).
        """
        request = LeaderRequest()
        self._writer.write(request.encode())
        await self._send()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, LeaderResponse):
            raise ProtocolError(f"Expected LeaderResponse, got {type(response).__name__}")

        return response.node_id, response.address

    async def open_database(self, name: str, flags: int = 0, vfs: str = "") -> int:
        """Open a database.

        Returns the database ID.
        """
        request = OpenRequest(name=name, flags=flags, vfs=vfs)
        self._writer.write(request.encode())
        await self._send()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, DbResponse):
            raise ProtocolError(f"Expected DbResponse, got {type(response).__name__}")

        return response.db_id

    async def prepare(self, db_id: int, sql: str) -> tuple[int, int]:
        """Prepare a SQL statement.

        Returns (stmt_id, num_params).
        """
        request = PrepareRequest(db_id=db_id, sql=sql)
        self._writer.write(request.encode())
        await self._send()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, StmtResponse):
            raise ProtocolError(f"Expected StmtResponse, got {type(response).__name__}")

        return response.stmt_id, response.num_params

    async def finalize(self, db_id: int, stmt_id: int) -> None:
        """Finalize (close) a prepared statement."""
        request = FinalizeRequest(db_id=db_id, stmt_id=stmt_id)
        self._writer.write(request.encode())
        await self._send()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, EmptyResponse):
            raise ProtocolError(f"Expected EmptyResponse, got {type(response).__name__}")

    async def exec_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[int, int]:
        """Execute SQL directly.

        Returns (last_insert_id, rows_affected). For multi-statement SQL
        (semicolon-separated), the server aggregates internally and returns
        a single RESULT with sqlite3_changes() of the last statement only —
        rows_affected is NOT a sum across statements.
        """
        request = ExecSqlRequest(db_id=db_id, sql=sql, params=params if params is not None else [])
        self._writer.write(request.encode())
        await self._send()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, ResultResponse):
            raise ProtocolError(f"Expected ResultResponse, got {type(response).__name__}")

        return response.last_insert_id, response.rows_affected

    async def query_sql_typed(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[Any]]]:
        """Execute a query and return (column_names, column_types, rows).

        column_types are the wire-level ``ValueType`` integer tags from the
        first response frame — what DBAPI cursor.description maps into
        ``type_code``.
        """
        request = QuerySqlRequest(db_id=db_id, sql=sql, params=params if params is not None else [])
        self._writer.write(request.encode())
        await self._send()

        deadline = self._operation_deadline()
        response = await self._read_response(deadline=deadline)
        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)
        if not isinstance(response, RowsResponse):
            raise ProtocolError(f"Expected RowsResponse, got {type(response).__name__}")

        column_names = list(response.column_names)
        column_types = [int(t) for t in response.column_types]
        all_rows = list(response.rows)
        while response.has_more:
            next_response = await self._read_continuation(deadline=deadline)
            all_rows.extend(next_response.rows)
            response = next_response
        return column_names, column_types, all_rows

    async def query_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query directly.

        Returns (column_names, rows). Multi-statement SELECT is rejected
        by the server with OperationalError(SQLITE_ERROR, "nonempty
        statement tail") — there are no additional result sets to drain.
        Use :meth:`query_sql_typed` to also get per-column ``ValueType``
        tags.
        """
        request = QuerySqlRequest(db_id=db_id, sql=sql, params=params if params is not None else [])
        self._writer.write(request.encode())
        await self._send()

        # Single deadline spans the initial response plus every continuation
        # frame; otherwise a server that split a reply into N frames could
        # legitimately take N * self._timeout to complete.
        deadline = self._operation_deadline()
        response = await self._read_response(deadline=deadline)

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, RowsResponse):
            raise ProtocolError(f"Expected RowsResponse, got {type(response).__name__}")

        column_names = response.column_names

        # Handle multi-part responses via decode_continuation(),
        # which decodes each continuation frame using the same layout
        # as the initial frame (column_count + column_names + rows +
        # marker), matching the C dqlite server's wire format.
        all_rows = list(response.rows)
        while response.has_more:
            next_response = await self._read_continuation(deadline=deadline)
            if not next_response.rows and next_response.has_more:
                # Server claimed "more coming" but delivered zero rows in a
                # continuation frame. That would spin forever (known
                # pathological case: column header larger than the server's
                # page buffer). Bail out instead of livelocking.
                raise ProtocolError(
                    "ROWS continuation made no progress: frame had 0 rows and has_more=True"
                )
            all_rows.extend(next_response.rows)
            response = next_response

        return column_names, all_rows

    async def _send(self) -> None:
        """Drain the writer, wrapping transport errors as DqliteConnectionError."""
        try:
            await self._writer.drain()
        except (ConnectionError, OSError, RuntimeError) as e:
            raise DqliteConnectionError(f"Write failed: {e}") from e

    async def _read_data(self, deadline: float | None = None) -> bytes:
        """Read a chunk from the stream, bounded by a per-operation deadline.

        If ``deadline`` is set (monotonic time), the per-chunk timeout is
        capped by the remaining budget — a slow-drip server that returned
        just under the per-read timeout on every chunk used to be able to
        keep a call alive indefinitely.

        Transport errors (ConnectionResetError, BrokenPipeError, OSError,
        RuntimeError("Transport is closed")) are wrapped in
        DqliteConnectionError to match the write-path behaviour.
        """
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise DqliteConnectionError(f"Operation exceeded {self._timeout}s deadline")
            timeout = min(remaining, self._timeout)
        else:
            timeout = self._timeout
        try:
            data = await asyncio.wait_for(self._reader.read(4096), timeout=timeout)
        except TimeoutError:
            raise DqliteConnectionError(f"Server read timed out after {timeout:.1f}s") from None
        except (ConnectionError, OSError, RuntimeError) as e:
            raise DqliteConnectionError(f"Read failed: {e}") from e
        if not data:
            raise DqliteConnectionError("Connection closed by server")
        return data

    def _operation_deadline(self) -> float:
        """Deadline (monotonic seconds) for a single protocol operation."""
        return asyncio.get_running_loop().time() + self._timeout

    async def _read_continuation(self, deadline: float | None = None) -> RowsResponse:
        """Read and decode a ROWS continuation frame.

        If ``deadline`` is None, a fresh per-operation deadline is set;
        query_sql passes its own deadline so the budget spans every
        continuation frame, not each one individually.
        """
        if deadline is None:
            deadline = self._operation_deadline()
        try:
            while True:
                result = self._decoder.decode_continuation()
                if result is not None:
                    return result
                data = await self._read_data(deadline=deadline)
                self._decoder.feed(data)
        except _WireProtocolError as e:
            raise ProtocolError(f"Wire decode failed: {e}") from e

    async def _read_response(self, deadline: float | None = None) -> Message:
        """Read and decode the next response message.

        If ``deadline`` is None, a fresh per-operation deadline is set for
        this one response; callers that span multiple reads (e.g. query_sql
        across continuation frames) pass an externally-held deadline so
        the cumulative wall time is bounded.
        """
        if deadline is None:
            deadline = self._operation_deadline()
        try:
            while not self._decoder.has_message():
                data = await self._read_data(deadline=deadline)
                self._decoder.feed(data)

            message = self._decoder.decode()
        except _WireProtocolError as e:
            raise ProtocolError(f"Wire decode failed: {e}") from e

        if message is None:
            raise ProtocolError("Failed to decode message")

        return message

    def close(self) -> None:
        """Close the connection."""
        self._writer.close()

    async def wait_closed(self) -> None:
        """Wait for the connection to close."""
        await self._writer.wait_closed()
