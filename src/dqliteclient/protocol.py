"""Low-level protocol handler for dqlite."""

import asyncio
from collections.abc import Sequence
from typing import Any

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqlitewire import MessageDecoder, MessageEncoder
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
        self._encoder = MessageEncoder()
        self._decoder = MessageDecoder(is_request=False)
        self._client_id = 0
        self._heartbeat_timeout = 0
        self._timeout = timeout

    async def handshake(self, client_id: int = 0) -> int:
        """Perform protocol handshake.

        Returns the heartbeat timeout from server.
        """
        # Send protocol version + client registration together
        request = ClientRequest(client_id=client_id)
        self._writer.write(self._encoder.encode_handshake() + request.encode())
        await self._writer.drain()

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
        await self._writer.drain()

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
        await self._writer.drain()

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
        await self._writer.drain()

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
        await self._writer.drain()

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
        (semicolon-separated), returns the last statement's last_insert_id
        and the sum of all rows_affected, matching the Go client behavior.
        """
        request = ExecSqlRequest(db_id=db_id, sql=sql, params=params or [])
        self._writer.write(request.encode())
        await self._writer.drain()

        last_insert_id = 0
        rows_affected = 0

        while True:
            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(response.code, response.message)

            if not isinstance(response, ResultResponse):
                raise ProtocolError(
                    f"Expected ResultResponse, got {type(response).__name__}"
                )

            last_insert_id = response.last_insert_id
            rows_affected += response.rows_affected

            # Check for more results from multi-statement SQL
            if not self._decoder.has_message():
                break

        return last_insert_id, rows_affected

    async def query_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query directly.

        Returns (column_names, rows).
        """
        request = QuerySqlRequest(db_id=db_id, sql=sql, params=params or [])
        self._writer.write(request.encode())
        await self._writer.drain()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, RowsResponse):
            raise ProtocolError(f"Expected RowsResponse, got {type(response).__name__}")

        # Store column names from first response
        column_names = response.column_names

        # Handle multi-part responses via decode_continuation(),
        # which decodes each continuation frame using the same layout
        # as the initial frame (column_count + column_names + rows +
        # marker), matching the C dqlite server's wire format.
        all_rows = list(response.rows)
        while response.has_more:
            next_response = await self._read_continuation()
            all_rows.extend(next_response.rows)
            response = next_response

        return column_names, all_rows

    async def _read_data(self) -> bytes:
        """Read data from the stream with timeout."""
        try:
            data = await asyncio.wait_for(
                self._reader.read(4096), timeout=self._timeout
            )
        except TimeoutError:
            raise DqliteConnectionError(
                f"Server read timed out after {self._timeout}s"
            ) from None
        if not data:
            raise DqliteConnectionError("Connection closed by server")
        return data

    async def _read_continuation(self) -> RowsResponse:
        """Read and decode a ROWS continuation frame."""
        while True:
            result = self._decoder.decode_continuation()
            if result is not None:
                return result
            data = await self._read_data()
            self._decoder.feed(data)

    async def _read_response(self) -> Message:
        """Read and decode the next response message."""
        while not self._decoder.has_message():
            data = await self._read_data()
            self._decoder.feed(data)

        message = self._decoder.decode()
        if message is None:
            raise ProtocolError("Failed to decode message")

        return message

    def close(self) -> None:
        """Close the connection."""
        self._writer.close()

    async def wait_closed(self) -> None:
        """Wait for the connection to close."""
        await self._writer.wait_closed()
