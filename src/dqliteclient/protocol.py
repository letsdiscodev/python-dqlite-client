"""Low-level protocol handler for dqlite."""

import asyncio
from typing import Any

from dqliteclient.exceptions import ConnectionError, OperationalError, ProtocolError
from dqlitewire import MessageDecoder, MessageEncoder, ReadBuffer
from dqlitewire.messages import (
    ClientRequest,
    DbResponse,
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
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._encoder = MessageEncoder()
        self._decoder = MessageDecoder(is_request=False)
        self._buffer = ReadBuffer()
        self._client_id = 0
        self._heartbeat_timeout = 0

    async def handshake(self, client_id: int = 0) -> int:
        """Perform protocol handshake.

        Returns the heartbeat timeout from server.
        """
        # Send protocol version
        self._writer.write(self._encoder.encode_handshake())
        await self._writer.drain()

        # Send client registration
        request = ClientRequest(client_id=client_id)
        self._writer.write(request.encode())
        await self._writer.drain()

        # Read welcome response
        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise ProtocolError(f"Handshake failed: {response.message}")

        if not isinstance(response, WelcomeResponse):
            raise ProtocolError(f"Expected WelcomeResponse, got {type(response).__name__}")

        self._client_id = client_id
        self._heartbeat_timeout = response.heartbeat_timeout
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

    async def exec_sql(
        self, db_id: int, sql: str, params: list[Any] | None = None
    ) -> tuple[int, int]:
        """Execute SQL directly.

        Returns (last_insert_id, rows_affected).
        """
        request = ExecSqlRequest(db_id=db_id, sql=sql, params=params or [])
        self._writer.write(request.encode())
        await self._writer.drain()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, response.message)

        if not isinstance(response, ResultResponse):
            raise ProtocolError(f"Expected ResultResponse, got {type(response).__name__}")

        return response.last_insert_id, response.rows_affected

    async def query_sql(
        self, db_id: int, sql: str, params: list[Any] | None = None
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

        # Handle multi-part responses
        all_rows = list(response.rows)
        while response.has_more:
            next_response = await self._read_response()
            if isinstance(next_response, RowsResponse):
                all_rows.extend(next_response.rows)
                response = next_response
            else:
                break

        return column_names, all_rows

    async def _read_response(self) -> Message:
        """Read and decode the next response message."""
        while not self._decoder.has_message():
            data = await self._reader.read(4096)
            if not data:
                raise ConnectionError("Connection closed by server")
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
