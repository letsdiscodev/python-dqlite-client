"""One wire session to a dqlite node: framing, deadlines and row streaming."""

import asyncio
import logging
import secrets
from collections.abc import Sequence
from typing import Any, Final, NoReturn, TypeIs, cast

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES,
    DEFAULT_MAX_TOTAL_ROWS,
    WIRE_DECODE_FAILED_PREFIX,
    Message,
    MessageDecoder,
    MessageEncoder,
    NodeInfo,
    NodeRole,
    ReadBuffer,
    sanitize_for_log,
    sanitize_server_text,
)
from dqlitewire import ProtocolError as WireProtocolError
from dqlitewire import ServerFailure as WireServerFailure
from dqlitewire.constants import ResponseType
from dqlitewire.messages import (
    AddRequest,
    AssignRequest,
    ClientRequest,
    ClusterRequest,
    DbResponse,
    DescribeRequest,
    DumpRequest,
    EmptyResponse,
    ExecSqlRequest,
    FailureResponse,
    FilesResponse,
    LeaderRequest,
    LeaderResponse,
    MetadataResponse,
    OpenRequest,
    QuerySqlRequest,
    RemoveRequest,
    ResultResponse,
    RowsResponse,
    ServersResponse,
    TransferRequest,
    WeightRequest,
    WelcomeResponse,
)

__all__ = ["DqliteProtocol", "validate_positive_int_or_none"]

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGE_SIZE: Final[int] = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE

_READ_CHUNK_SIZE: Final[int] = 4096
# Upper bound on how far a trusted server heartbeat may widen the read deadline.
_HEARTBEAT_READ_TIMEOUT_CAP_SECONDS: Final[float] = 300.0
# Frames at or above these sizes are encoded / decoded on a worker thread.
_ENCODE_OFFLOAD_THRESHOLD: Final[int] = 256 * 1024
_DECODE_OFFLOAD_THRESHOLD: Final[int] = 256 * 1024
_MAX_ERROR_MESSAGE_SNIPPET: Final[int] = 200


def _is_int_not_bool(value: object) -> TypeIs[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_positive_int_or_none(value: int | None, name: str) -> int | None:
    """``None`` disables the cap; otherwise an int >= 1."""
    if value is None:
        return None
    if not _is_int_not_bool(value):
        raise TypeError(f"{name} must be int or None, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0 or None, got {value}")
    return value


def _truncate_error(message: str) -> str:
    safe = sanitize_server_text(message)
    if len(safe) <= _MAX_ERROR_MESSAGE_SNIPPET:
        return safe
    overflow = len(safe) - _MAX_ERROR_MESSAGE_SNIPPET
    return safe[:_MAX_ERROR_MESSAGE_SNIPPET] + f"... [truncated, {overflow} chars]"


def _failure_message(message: str, addr_suffix: str) -> str:
    body = message if message.strip() else "(no diagnostic from server)"
    return body + addr_suffix


def _estimate_request_body_size(request: object) -> int:
    """Upper bound of the encoded SQL text and parameters (UTF-8 may expand 4x)."""
    total = 0
    sql = getattr(request, "sql", None)
    if isinstance(sql, str):
        total += len(sql) * 4
    for value in getattr(request, "params", None) or ():
        if isinstance(value, bytes | bytearray | memoryview):
            total += len(value)
        elif isinstance(value, str):
            total += len(value) * 4
    return total


class DqliteProtocol:
    """Request/response plumbing over one socket. One RPC at a time (the lock
    serialises callers); a failed or cancelled RPC leaves the stream position unknown,
    so owners close the protocol rather than reuse it."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        timeout: float = 10.0,
        max_total_rows: int | None = DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        address: str | None = None,
        max_message_size: int | None = None,
    ) -> None:
        if max_message_size is None:
            max_message_size = DEFAULT_MAX_MESSAGE_SIZE
        if not _is_int_not_bool(max_message_size):
            raise TypeError(
                f"max_message_size must be int or None, got {type(max_message_size).__name__}"
            )
        if max_message_size < 1:
            raise ValueError(f"max_message_size must be >= 1, got {max_message_size}")
        self._reader = reader
        self._writer = writer
        self._decoder = MessageDecoder(
            is_request=False,
            max_total_rows=max_total_rows,
            max_continuation_frames=max_continuation_frames,
            max_message_size=max_message_size,
        )
        self._encoder = MessageEncoder(max_message_size=max_message_size)
        self._max_message_size = max_message_size
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._timeout = timeout
        self._read_timeout = timeout
        self._trust_server_heartbeat = trust_server_heartbeat
        self._address = address
        self._client_id = 0
        self._heartbeat_timeout = 0
        self._lock = asyncio.Lock()

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"cannot pickle {type(self).__name__!r}: it owns a live socket")

    @property
    def is_alive(self) -> bool:
        """False once the transport is closing, the peer sent EOF, or the decoder lost sync."""
        return (
            not self._writer.is_closing()
            and not self._reader.at_eof()
            and not self._decoder.is_poisoned
        )

    # -- handshake ---------------------------------------------------------------------

    async def negotiate_protocol_only(self) -> None:
        """Send the version bytes only; the peer's first frame stays buffered for the
        caller's next RPC. Enough for leader probes, which need no client registration."""
        await self._send(self._encoder.encode_handshake())

    async def handshake(self, client_id: int | None = None) -> int:
        """Version exchange plus client registration in one write; returns the server's
        heartbeat timeout (ms)."""
        if client_id is None:
            client_id = secrets.randbits(63) or 1  # distinguishes sessions in server logs
        self._client_id = client_id
        request = ClientRequest(client_id=client_id)
        await self._send(self._encoder.encode_handshake() + self._encoder.encode(request))
        response = await self._read_response()
        if isinstance(response, FailureResponse):
            raise OperationalError(
                f"Handshake failed: [{response.code}] {self._failure_text(response)}",
                response.code,
                raw_message=response.message,
            )
        if not isinstance(response, WelcomeResponse):
            raise ProtocolError(
                f"Expected WelcomeResponse, got {type(response).__name__}{self._addr_suffix()}"
            )
        self._heartbeat_timeout = response.heartbeat_timeout
        if self._trust_server_heartbeat and response.heartbeat_timeout > 0:
            heartbeat_seconds = response.heartbeat_timeout / 1000.0
            if heartbeat_seconds > _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS:
                logger.warning(
                    "handshake: server heartbeat %.2fs exceeds the %.2fs cap; clipping",
                    heartbeat_seconds,
                    _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS,
                )
                heartbeat_seconds = _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS
            self._read_timeout = max(self._read_timeout, heartbeat_seconds)
        return response.heartbeat_timeout

    # -- RPCs --------------------------------------------------------------------------

    async def _rpc[T: Message](self, request: Message, expected: type[T]) -> T:
        async with self._lock:
            await self._send_request(request)
            return self._expect(await self._read_response(), expected)

    def _expect[T: Message](self, response: Message, expected: type[T]) -> T:
        if isinstance(response, FailureResponse):
            self._raise_failure(response)
        if not isinstance(response, expected):
            raise ProtocolError(
                f"Expected {expected.__name__}, got {type(response).__name__}{self._addr_suffix()}"
            )
        return response

    def _raise_failure(self, response: FailureResponse) -> NoReturn:
        raise OperationalError(
            self._failure_text(response), response.code, raw_message=response.message
        )

    async def get_leader(self) -> tuple[int, str]:
        """``(node_id, address)``; ``(0, "")`` when no leader is known."""
        response = await self._rpc(LeaderRequest(), LeaderResponse)
        if response.node_id == 0 and response.address:
            raise ProtocolError(
                f"server returned address {sanitize_server_text(response.address)!r} "
                f"with node_id=0{self._addr_suffix()}"
            )
        return response.node_id, response.address

    async def cluster(self) -> list[NodeInfo]:
        return (await self._rpc(ClusterRequest(format=1), ServersResponse)).nodes

    async def add(self, node_id: int, address: str) -> None:
        """Add a node as a spare; promote it with :meth:`assign`. Leader only."""
        await self._rpc(AddRequest(node_id=node_id, address=address), EmptyResponse)

    async def assign(self, node_id: int, role: NodeRole) -> None:
        await self._rpc(AssignRequest(node_id=node_id, role=role), EmptyResponse)

    async def remove(self, node_id: int) -> None:
        await self._rpc(RemoveRequest(node_id=node_id), EmptyResponse)

    async def describe(self) -> MetadataResponse:
        """Failure domain and weight of the connected node."""
        return await self._rpc(DescribeRequest(format=0), MetadataResponse)

    async def weight(self, weight: int) -> None:
        await self._rpc(WeightRequest(weight=weight), EmptyResponse)

    async def transfer(self, target_node_id: int) -> None:
        """Ask the leader to hand over leadership; returns once the request is accepted."""
        await self._rpc(TransferRequest(target_node_id=target_node_id), EmptyResponse)

    async def open_database(self, name: str, flags: int = 0, vfs: str = "") -> int:
        response = await self._rpc(OpenRequest(name=name, flags=flags, vfs=vfs), DbResponse)
        if response.db_id != 0:  # the server always assigns 0 to a session's first database
            raise ProtocolError(
                f"{WIRE_DECODE_FAILED_PREFIX}: OPEN returned db_id={response.db_id}, "
                f"expected 0{self._addr_suffix()}"
            )
        return response.db_id

    async def exec_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[int, int]:
        """``(last_insert_id, rows_affected)``; for multi-statement SQL the last one's."""
        request = ExecSqlRequest(db_id=db_id, sql=sql, params=list(params or ()))
        response = await self._rpc(request, ResultResponse)
        return response.last_insert_id, response.rows_affected

    async def dump(self, database: str) -> dict[str, bytes]:
        """``{filename: bytes}`` for the database file and its WAL."""
        async with self._lock:
            await self._send_request(DumpRequest(name=database))
            frame = await self._read_frame_bytes()
            try:
                response = await asyncio.to_thread(self._decoder.decode_bytes, frame)
            except WireProtocolError as exc:
                raise ProtocolError(
                    f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {exc}"
                ) from exc
            return self._expect(response, FilesResponse).files

    async def query_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """``(column_names, rows)``."""
        async with self._lock:
            first, deadline = await self._send_query(db_id, sql, params)
            rows, _ = await self._drain_continuations(first, deadline)
            return first.column_names, rows

    async def query_sql_typed(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """``(column_names, column_types, row_types, rows)``: the first frame's column tags
        plus one type list per row, since SQLite types cells, not columns."""
        async with self._lock:
            first, deadline = await self._send_query(db_id, sql, params)
            rows, row_types = await self._drain_continuations(first, deadline)
            return list(first.column_names), [int(t) for t in first.column_types], row_types, rows

    async def _send_query(
        self, db_id: int, sql: str, params: Sequence[Any] | None
    ) -> tuple[RowsResponse, float]:
        await self._send_request(QuerySqlRequest(db_id=db_id, sql=sql, params=list(params or ())))
        deadline = self._operation_deadline()
        response = self._expect(await self._read_response(deadline), RowsResponse)
        return response, deadline

    async def _drain_continuations(
        self, first: RowsResponse, deadline: float
    ) -> tuple[list[list[Any]], list[list[int]]]:
        rows: list[list[Any]] = list(first.rows)
        row_types = cast("list[list[int]]", list(first.row_types))
        self._check_row_cap(len(rows))
        frames = 1
        response = first
        while response.has_more:
            if (
                self._max_continuation_frames is not None
                and frames >= self._max_continuation_frames
            ):
                raise ProtocolError(
                    f"Query exceeded max_continuation_frames cap "
                    f"({self._max_continuation_frames}){self._addr_suffix()}"
                )
            response = await self._read_continuation(deadline)
            frames += 1
            if not response.rows and response.has_more:
                raise ProtocolError(
                    f"ROWS continuation made no progress (0 rows, has_more=True)"
                    f"{self._addr_suffix()}"
                )
            self._check_row_cap(len(rows) + len(response.rows))
            rows.extend(response.rows)
            row_types.extend(cast("list[list[int]]", response.row_types))
            await asyncio.sleep(0)  # buffered frames would otherwise pin the loop
        return rows, row_types

    def _check_row_cap(self, total: int) -> None:
        if self._max_total_rows is not None and total > self._max_total_rows:
            raise ProtocolError(
                f"Query exceeded max_total_rows cap ({self._max_total_rows}); reduce the "
                f"result size or raise the cap{self._addr_suffix()}"
            )

    # -- framing -----------------------------------------------------------------------

    async def _send_request(self, request: Message) -> None:
        if _estimate_request_body_size(request) >= _ENCODE_OFFLOAD_THRESHOLD:
            frame = await asyncio.to_thread(self._encoder.encode, request)
        else:
            frame = self._encoder.encode(request)
        await self._send(frame)

    async def _send(self, frame: bytes) -> None:
        try:
            self._writer.write(frame)
            async with asyncio.timeout(self._timeout):
                await self._writer.drain()
        except TimeoutError as exc:
            raise DqliteConnectionError(
                f"Write timeout{self._addr_suffix()} after {self._timeout}s"
            ) from exc
        except (OSError, RuntimeError) as exc:  # RuntimeError: write on a closed transport
            raise DqliteConnectionError(f"Write failed{self._addr_suffix()}: {exc}") from exc

    async def _read_data(self, deadline: float) -> bytes:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise DqliteConnectionError(
                f"Operation{self._addr_suffix()} exceeded deadline by {-remaining:.3f}s"
            )
        timeout = min(remaining, self._read_timeout)
        try:
            async with asyncio.timeout(timeout):
                data = await self._reader.read(_READ_CHUNK_SIZE)
        except TimeoutError as exc:
            raise DqliteConnectionError(
                f"Server read{self._addr_suffix()} timed out after {timeout:.1f}s"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise DqliteConnectionError(f"Read failed{self._addr_suffix()}: {exc}") from exc
        if not data:
            raise DqliteConnectionError(f"Connection closed by server{self._addr_suffix()}")
        return data

    async def _fill(self, deadline: float) -> None:
        """Read until a whole frame is buffered."""
        while not self._decoder.has_message():
            self._decoder.feed(await self._read_data(deadline))
            await asyncio.sleep(0)

    async def _read_response(self, deadline: float | None = None) -> Message:
        """Decode the next frame; a second buffered frame after a terminal response
        means the peer violated the protocol."""
        if deadline is None:
            deadline = self._operation_deadline()
        try:
            await self._fill(deadline)
            if self._decoder.pending_frame_size() >= _DECODE_OFFLOAD_THRESHOLD:
                message = await asyncio.to_thread(self._decoder.decode)
            else:
                message = self._decoder.decode()
        except WireProtocolError as exc:
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {exc}") from exc
        if message is None:
            raise ProtocolError(f"Failed to decode message{self._addr_suffix()}")
        if not isinstance(message, RowsResponse) and self._decoder.has_message():
            raise ProtocolError(
                f"Server emitted an extra response after {type(message).__name__}"
                f"{self._addr_suffix()}"
            )
        return message

    async def _read_continuation(self, deadline: float) -> RowsResponse:
        try:
            while True:
                if self._decoder.pending_frame_size() >= _DECODE_OFFLOAD_THRESHOLD:
                    result = await asyncio.to_thread(self._decoder.decode_continuation)
                else:
                    result = self._decoder.decode_continuation()
                if isinstance(result, EmptyResponse):
                    raise ProtocolError(
                        f"Query was interrupted server-side mid-stream{self._addr_suffix()}"
                    )
                if result is not None:
                    return result
                self._decoder.feed(await self._read_data(deadline))
                await asyncio.sleep(0)
        except WireServerFailure as exc:
            raise OperationalError(
                _failure_message(_truncate_error(exc.message), self._addr_suffix()),
                exc.code,
                raw_message=exc.message,
            ) from exc
        except WireProtocolError as exc:
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {exc}") from exc

    async def _read_frame_bytes(self) -> bytes:
        """The next frame undecoded (for off-loop decoding of large payloads)."""
        try:
            await self._fill(self._operation_deadline())
            frame = self._decoder.take_frame()
        except WireProtocolError as exc:
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {exc}") from exc
        if frame is None:
            raise ProtocolError(f"Failed to read message bytes{self._addr_suffix()}")
        if frame[4] != ResponseType.ROWS and self._decoder.has_message():
            raise ProtocolError(
                f"Server emitted an extra response after message type {frame[4]}"
                f"{self._addr_suffix()}"
            )
        return frame

    def _operation_deadline(self) -> float:
        return asyncio.get_running_loop().time() + self._read_timeout

    def _addr_suffix(self) -> str:
        return f" to {sanitize_for_log(self._address)}" if self._address else ""

    def _failure_text(self, response: FailureResponse) -> str:
        return _failure_message(_truncate_error(response.message), self._addr_suffix())

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        self._writer.close()

    async def wait_closed(self) -> None:
        await self._writer.wait_closed()
