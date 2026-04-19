"""Low-level protocol handler for dqlite."""

import asyncio
import logging
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

logger = logging.getLogger(__name__)

# Socket read buffer size. 4 KiB balances syscall overhead for typical
# request/response payloads against latency for small wire messages.
_READ_CHUNK_SIZE = 4096


def _validate_positive_int_or_none(value: int | None, name: str) -> int | None:
    """Shared validation for positive-int-or-None parameters.

    Used for both ``max_total_rows`` and ``max_continuation_frames``.
    None disables the cap; any int value must be > 0.
    """
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int or None, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0 or None, got {value}")
    return value


class DqliteProtocol:
    """Low-level protocol handler for a single dqlite connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        timeout: float = 15.0,
        max_total_rows: int | None = 10_000_000,
        max_continuation_frames: int | None = 100_000,
        trust_server_heartbeat: bool = False,
        address: str | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._decoder = MessageDecoder(is_request=False)
        self._client_id = 0
        self._heartbeat_timeout = 0
        self._timeout = timeout
        # Diagnostic-only peer address. Embedded into timeout and
        # decode-error messages so operators can tell from the
        # exception alone which node a hung probe / mangled frame came
        # from; callers without the address in scope may omit it.
        self._address = address
        # Cumulative cap across continuation frames for a single query.
        # A hostile or buggy server can drip-feed 1-row-per-frame inside
        # the per-operation deadline; without a cumulative cap, clients
        # could legitimately allocate hundreds of millions of rows over
        # the full deadline. None disables the cap.
        self._max_total_rows = _validate_positive_int_or_none(max_total_rows, "max_total_rows")
        # Per-query frame cap. Complements max_total_rows: a server
        # sending 10M 1-row frames to reach the row cap would still
        # burn 10M × decode-cost of Python work; the frame cap bounds
        # that at ~100k iterations.
        self._max_continuation_frames = _validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        # When True, the client honors the server-advertised heartbeat
        # timeout to adjust its per-read deadline (subject to the 300 s
        # hard cap). When False (default), the server value is recorded
        # for diagnostics only and the operator-configured ``timeout``
        # is authoritative. Opt-in protects operators whose timeout is
        # a latency-SLO boundary from server-induced amplification.
        self._trust_server_heartbeat = trust_server_heartbeat

    async def handshake(self, client_id: int | None = None) -> int:
        """Perform protocol handshake.

        If ``client_id`` is not provided, a random non-zero 63-bit id is
        generated so each connection is distinguishable in server logs,
        traces, and per-client metrics. Returns the heartbeat timeout
        from the server.
        """
        if client_id is None:
            # Deliberate divergence from go-dqlite: Go leaves the default
            # client_id; we randomize so each connection is distinguishable
            # in server logs, traces, and per-client metrics. 63 bits
            # avoids sign-extension pitfalls if an intermediate layer
            # treats the id as int64. The ``or 1`` guards against the
            # (astronomically unlikely) all-zero draw.
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
        # Use the server-advertised heartbeat only when explicitly
        # trusted. Previously we always widened ``self._timeout`` up
        # to 300 s based on the server value, which let a hostile
        # server amplify the operator's configured timeout up to 30×.
        # Default is now opt-out: the server value is
        # recorded for diagnostics but does not change the per-read
        # deadline.
        if self._trust_server_heartbeat and response.heartbeat_timeout > 0:
            heartbeat_seconds = response.heartbeat_timeout / 1000.0
            # Cap to prevent a malicious/buggy server from disabling timeouts
            new_timeout = max(self._timeout, min(heartbeat_seconds, 300.0))
            if new_timeout != self._timeout:
                # Security-relevant opt-in: surface the actual widening
                # at DEBUG so an operator who flipped the knob can
                # confirm it took effect.
                logger.debug(
                    "handshake: widened per-read timeout %.2fs -> %.2fs (server heartbeat=%.2fs)",
                    self._timeout,
                    new_timeout,
                    heartbeat_seconds,
                )
                self._timeout = new_timeout
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

    async def _send_query(
        self, db_id: int, sql: str, params: Sequence[Any] | None
    ) -> tuple["RowsResponse", float]:
        """Send a QUERY_SQL request and return the first RowsResponse + deadline.

        Raises OperationalError for server FailureResponse and ProtocolError
        for any other unexpected message type.
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
        return response, deadline

    async def _drain_continuations(
        self, initial: "RowsResponse", deadline: float
    ) -> list[list[Any]]:
        """Drain all continuation frames, enforcing the progress + total-row caps.

        Returns the full list of rows (initial frame first). A
        continuation claiming more rows with zero delivered, or a
        cumulative row count exceeding ``max_total_rows``, raises
        ProtocolError.
        """
        all_rows = list(initial.rows)
        response = initial
        frames = 1  # the initial frame counts
        while response.has_more:
            next_response = await self._read_continuation(deadline=deadline)
            frames += 1
            if not next_response.rows and next_response.has_more:
                raise ProtocolError(
                    "ROWS continuation made no progress: frame had 0 rows and has_more=True"
                )
            if self._max_continuation_frames is not None and frames > self._max_continuation_frames:
                # Per-frame cap complements max_total_rows: a
                # slow-drip server sending 1-row-per-frame would
                # otherwise pin a client CPU with O(n) iterations of
                # decode work, where n is max_total_rows.
                raise ProtocolError(
                    f"Query exceeded max_continuation_frames cap "
                    f"({self._max_continuation_frames}); server may be "
                    f"slow-dripping rows."
                )
            if self._max_total_rows is not None and (
                len(all_rows) + len(next_response.rows) > self._max_total_rows
            ):
                raise ProtocolError(
                    f"Query exceeded max_total_rows cap ({self._max_total_rows}); "
                    f"reduce result size or raise the cap on the connection/pool."
                )
            all_rows.extend(next_response.rows)
            response = next_response
        return all_rows

    async def query_sql_typed(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[Any]]]:
        """Execute a query and return (column_names, column_types, rows).

        column_types are the wire-level ``ValueType`` integer tags from the
        first response frame — what DBAPI cursor.description maps into
        ``type_code``.

        Atomicity: a mid-stream server failure or an unexpected message
        type raises before any rows are returned to the caller; the
        local row list is discarded. The connection is invalidated so
        callers don't accidentally reuse it with torn protocol state.
        """
        response, deadline = await self._send_query(db_id, sql, params)
        column_names = list(response.column_names)
        column_types = [int(t) for t in response.column_types]
        all_rows = await self._drain_continuations(response, deadline)
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
        response, deadline = await self._send_query(db_id, sql, params)
        column_names = response.column_names
        all_rows = await self._drain_continuations(response, deadline)

        return column_names, all_rows

    async def _send(self) -> None:
        """Drain the writer, wrapping transport errors as DqliteConnectionError."""
        try:
            await self._writer.drain()
        except (ConnectionError, OSError, RuntimeError) as e:
            raise DqliteConnectionError(f"Write failed{self._addr_suffix()}: {e}") from e

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
                raise DqliteConnectionError(
                    f"Operation{self._addr_suffix()} exceeded {self._timeout}s deadline"
                )
            timeout = min(remaining, self._timeout)
        else:
            timeout = self._timeout
        try:
            data = await asyncio.wait_for(self._reader.read(_READ_CHUNK_SIZE), timeout=timeout)
        except TimeoutError as e:
            raise DqliteConnectionError(
                f"Server read{self._addr_suffix()} timed out after {timeout:.1f}s"
            ) from e
        except (ConnectionError, OSError, RuntimeError) as e:
            raise DqliteConnectionError(f"Read failed{self._addr_suffix()}: {e}") from e
        if not data:
            raise DqliteConnectionError(f"Connection closed by server{self._addr_suffix()}")
        return data

    def _addr_suffix(self) -> str:
        """Render the peer address as a trailing ``" to <addr>"`` fragment.

        Returns an empty string when the address is unknown — keeping
        error messages clean for callers that don't thread it in.
        """
        return f" to {self._address}" if self._address else ""

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
            raise ProtocolError(f"Wire decode failed{self._addr_suffix()}: {e}") from e

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
            raise ProtocolError(f"Wire decode failed{self._addr_suffix()}: {e}") from e

        if message is None:
            raise ProtocolError(f"Failed to decode message{self._addr_suffix()}")

        return message

    def close(self) -> None:
        """Close the connection."""
        self._writer.close()

    async def wait_closed(self) -> None:
        """Wait for the connection to close."""
        await self._writer.wait_closed()
