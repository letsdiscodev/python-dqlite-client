"""Low-level protocol handler for dqlite."""

import asyncio
import logging
import secrets
from collections.abc import Sequence
from typing import Any, Final, NoReturn, TypeIs, cast

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)
from dqlitewire import (
    WIRE_DECODE_FAILED_PREFIX,
    Message,
    MessageDecoder,
    MessageEncoder,
    NodeInfo,
    NodeRole,
    ReadBuffer,
)
from dqlitewire import ProtocolError as _WireProtocolError
from dqlitewire import ServerFailure as _WireServerFailure
from dqlitewire import (
    sanitize_for_log as _sanitize_for_log,
)
from dqlitewire import (
    sanitize_server_text as _sanitize_display_text,
)
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
    FinalizeRequest,
    InterruptRequest,
    LeaderRequest,
    LeaderResponse,
    MetadataResponse,
    OpenRequest,
    PrepareRequest,
    QuerySqlRequest,
    RemoveRequest,
    ResultResponse,
    RowsResponse,
    ServersResponse,
    StmtResponse,
    TransferRequest,
    WeightRequest,
    WelcomeResponse,
)

__all__ = ["DqliteProtocol", "validate_positive_int_or_none"]

logger = logging.getLogger(__name__)

_READ_CHUNK_SIZE: Final[int] = 4096

# Caps how far a server-advertised heartbeat can widen the per-read deadline
# under ``trust_server_heartbeat``, so a hostile server cannot disable read
# timeouts. Mirrored in the ``trust_server_heartbeat`` docstrings on
# ``DqliteProtocol`` / ``DqliteConnection`` and the ``connect`` / ``create_pool``
# docstrings in ``__init__.py``.
_HEARTBEAT_READ_TIMEOUT_CAP_SECONDS: Final[float] = 300.0

# Requests whose projected encoded body size reaches this encode on a worker
# thread so a multi-MiB param/SQL memcpy does not freeze the event loop.
_ENCODE_OFFLOAD_THRESHOLD: Final[int] = 256 * 1024

# Frames at/above this wire size decode on a worker thread so the per-row walk
# does not freeze the loop. See cancel-orphan note: an outer cancel cannot stop a
# running ``to_thread`` worker, but it lands on an abandoned decoder after
# ``_invalidate`` nulls ``_protocol`` — bounded resource use, no correctness defect.
_DECODE_OFFLOAD_THRESHOLD: Final[int] = 256 * 1024


def _decode_dump_response_sync(frame_bytes: bytes, decoder: MessageDecoder) -> Message:
    """Sync body of the ``dump`` off-loop decode hop (run via to_thread)."""
    return decoder.decode_bytes(frame_bytes)


def _estimate_request_body_size(request: object) -> int:
    """Cheap loop-thread pre-estimate of the encoded body size.

    Sums the ``sql`` text field and ``params``; returns 0 for requests with
    neither (keeping admin/handshake RPCs in-loop). ``len * 4`` upper-bounds
    UTF-8 expansion: over-estimating only offloads earlier, under would not.
    """
    total = 0
    sql = getattr(request, "sql", None)
    if isinstance(sql, str):
        total += len(sql) * 4
    params = getattr(request, "params", None)
    if params:
        for value in params:
            if isinstance(value, (bytes, bytearray, memoryview)):
                total += len(value)
            elif isinstance(value, str):
                total += len(value) * 4
            # Numeric/bool/None params are < 16 bytes each; negligible.
    return total


# Default inbound frame-size cap, re-exported from the wire layer (64 MiB) so
# the propagation chain has a single source of truth.
DEFAULT_MAX_MESSAGE_SIZE: Final[int] = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE


def _failure_message(message: str, addr_suffix: str) -> str:
    """Render a FailureResponse body, substituting a placeholder for empty text
    so log scraping has something to grep instead of ``"[1] "``."""
    body = message if message.strip() else "(no diagnostic from server)"
    return body + addr_suffix


def _is_int_not_bool(v: object) -> TypeIs[int]:
    """True for a genuine int; rejects bool (an int subclass) intentionally."""
    return isinstance(v, int) and not isinstance(v, bool)


def validate_positive_int_or_none(value: int | None, name: str) -> int | None:
    """Validate a positive-int-or-None parameter; None disables the cap.

    Public so downstream packages can reuse it without reaching into privates.
    """
    if value is None:
        return None
    if not _is_int_not_bool(value):
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
        timeout: float = 10.0,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        address: str | None = None,
        max_message_size: int | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        # None falls back to the wire-layer default (64 MiB).
        effective_max_message_size = (
            max_message_size if max_message_size is not None else DEFAULT_MAX_MESSAGE_SIZE
        )
        if not isinstance(effective_max_message_size, int) or isinstance(
            effective_max_message_size, bool
        ):
            raise TypeError(
                f"max_message_size must be int or None, "
                f"got {type(effective_max_message_size).__name__}"
            )
        if effective_max_message_size < 1:
            raise ValueError(f"max_message_size must be >= 1, got {effective_max_message_size}")
        self._decoder = MessageDecoder(
            is_request=False,
            max_total_rows=max_total_rows,
            max_continuation_frames=max_continuation_frames,
            max_message_size=effective_max_message_size,
        )
        # Symmetric outbound cap: reject oversized frames locally before the
        # writer rather than after a round-trip. Python-only defence-in-depth.
        self._encoder = MessageEncoder(max_message_size=effective_max_message_size)
        self._max_message_size = effective_max_message_size
        self._client_id = 0
        self._heartbeat_timeout = 0
        self._timeout = timeout
        # Per-read deadline; ``trust_server_heartbeat`` may widen this (up to
        # 300 s) but never ``self._timeout``, so the write SLO stays pinned.
        self._read_timeout = timeout
        # Diagnostic-only peer address, embedded into error messages.
        self._address = address
        # Cumulative row cap across a query's continuation frames so a server
        # drip-feeding 1 row/frame within the deadline cannot exhaust memory.
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        # Frame cap complements max_total_rows: bounds per-frame decode work
        # against a server sending many tiny frames to reach the row cap.
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._trust_server_heartbeat = trust_server_heartbeat
        # Serialise wire-touching RPCs: the dqlite server does not support
        # concurrent requests on one connection (mirrors go-dqlite's
        # Protocol.mu). In-tree callers hold ``_in_use`` one layer up; this
        # closes the gap for third-party callers sharing an instance.
        # ``_send`` / ``_read_*`` run inside locked methods; connect-time
        # ``handshake`` / ``negotiate_protocol_only`` intentionally skip it.
        self._lock = asyncio.Lock()

    def __reduce__(self) -> NoReturn:
        # Wraps loop-bound streams + stateful decoder; raise a clear TypeError
        # rather than leaking the underlying StreamReader pickle failure.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — wraps "
            f"loop-bound StreamReader / StreamWriter and a stateful "
            f"MessageDecoder; reconstruct from a fresh wire handshake "
            f"in the target process instead."
        )

    @property
    def is_wire_coherent(self) -> bool:
        """True if the codec-layer decoder buffer has not been poisoned.

        Reflects ONLY ``MessageDecoder.is_poisoned``, not client-layer
        ProtocolError raises (db_id mismatch, extra frame, drain caps) that
        leave coherent bytes but a desynchronised next-request boundary.
        A hint, not a liveness contract — a CancelledError could poison the
        buffer before the next op, so a True return is not "healthy". Used as
        a short-circuit by the pool reset path only; other callers classify
        wire desync through the ProtocolError exception chain.
        """
        return not self._decoder.is_poisoned

    def _compute_pending_size(self) -> int:
        """Best-effort decoder header peek for the offload threshold; 0 on any
        stub/MagicMock decoder whose non-int arithmetic would raise."""
        from dqlitewire.constants import WORD_SIZE

        try:
            pending = self._decoder._buffer.peek_header()
            pending_size = 8 + pending[0] * WORD_SIZE if pending is not None else 0
            if not isinstance(pending_size, int):
                pending_size = 0
        except (AttributeError, TypeError, IndexError):
            pending_size = 0
        return pending_size

    async def negotiate_protocol_only(self) -> None:
        """Probe-only handshake: write the version bytes with no ClientRequest.

        Leader probes can skip registration (``handle_leader`` needs no
        client_id), avoiding a per-client server slot. DO NOT use for
        connections issuing real queries — those need :meth:`handshake`.
        """
        await self._send(self._encoder.encode_handshake())

    async def handshake(self, client_id: int | None = None) -> int:
        """Perform protocol handshake; return the server's heartbeat timeout.

        Bundles the version exchange and ClientRequest into a single write so
        the kernel packs them into one TCP segment on the connect hot path.
        """
        if client_id is None:
            # Diverges from go-dqlite (always id=0). Randomise so each
            # connection is distinguishable in server logs/metrics. 63 bits
            # avoids int64 sign-extension; ``or 1`` keeps a ``client_id != 0``
            # filter reliable. ``secrets`` (not ``random``) because the PRNG
            # state is process-global and not fork-aware.
            client_id = secrets.randbits(63) or 1
        request = ClientRequest(client_id=client_id)
        # Record the id before the write so the FailureResponse arm can report
        # it: the server allocates its gateway slot (reclaimed on TCP close)
        # before composing any response.
        self._client_id = client_id
        await self._send(self._encoder.encode_handshake() + self._encoder.encode(request))

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            # OperationalError (not ProtocolError) so the server-supplied
            # SQLite code is preserved as an attribute for downstream
            # classifiers. ``raw_message`` keeps the verbatim peer text;
            # ``_failure_text`` pre-truncates so do NOT re-wrap in
            # ``_truncate_error`` (it would strip the addr suffix).
            raise OperationalError(
                f"Handshake failed (server-side client slot may be allocated "
                f"as id={client_id}; reclaimed on TCP close): "
                f"[{response.code}] {self._failure_text(response)}",
                response.code,
                raw_message=response.message,
            )

        if not isinstance(response, WelcomeResponse):
            raise ProtocolError(
                f"Expected WelcomeResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

        self._heartbeat_timeout = response.heartbeat_timeout
        if response.heartbeat_timeout == 0:
            # config.c defaults to 15000 and never emits 0, so 0 means a
            # misconfigured peer or non-conforming server.
            logger.debug(
                "handshake: server advertised heartbeat=0 (semantically "
                "ambiguous per wire spec; widening disabled)"
            )
        # Widen only when explicitly trusted; otherwise a hostile server could
        # amplify the configured timeout up to 30x.
        if self._trust_server_heartbeat and response.heartbeat_timeout > 0:
            heartbeat_seconds = response.heartbeat_timeout / 1000.0
            if heartbeat_seconds > _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS:
                logger.warning(
                    "handshake: server-advertised heartbeat %.2fs exceeds "
                    "client cap %.2fs; clipping",
                    heartbeat_seconds,
                    _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS,
                )
                heartbeat_seconds = _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS
            # Widen only the READ deadline; write-path self._timeout stays pinned.
            new_read_timeout = max(self._read_timeout, heartbeat_seconds)
            if new_read_timeout != self._read_timeout:
                logger.debug(
                    "handshake: widened per-read timeout %.2fs -> %.2fs (server heartbeat=%.2fs)",
                    self._read_timeout,
                    new_read_timeout,
                    heartbeat_seconds,
                )
                self._read_timeout = new_read_timeout
            else:
                # Server value <= configured deadline: opted-in but no widening
                # applied. Surface so the operator can recalibrate.
                logger.debug(
                    "handshake: trust_server_heartbeat=True but server "
                    "advertised %.2fs <= configured read timeout %.2fs; "
                    "no widening applied",
                    heartbeat_seconds,
                    self._read_timeout,
                )
        return response.heartbeat_timeout

    async def get_leader(self) -> tuple[int, str]:
        """Request leader information; return ``(node_id, address)``.

        ``(0, "")`` ("no leader known") passes through; ``(0, nonempty)`` is
        rejected as ProtocolError (raft_leader pairs id+address). ``(N, "")``
        is left for the cluster wrappers to log with per-address context.
        """
        async with self._lock:
            request = LeaderRequest()
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, LeaderResponse):
                raise ProtocolError(
                    f"Expected LeaderResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            if response.node_id == 0 and response.address:
                # raft_leader never pairs node_id=0 with an address; reject.
                raise ProtocolError(
                    f"server returned address "
                    f"{_sanitize_display_text(response.address)!r} "
                    f"with node_id=0; expected both or neither"
                    f"{self._addr_suffix()}"
                )

            return response.node_id, response.address

    async def cluster(self) -> list[NodeInfo]:
        """Request the cluster's node list (V1: id + address + role).

        Any node can answer (the view is replicated), but the freshest view
        comes from the leader.
        """
        async with self._lock:
            request = ClusterRequest(format=1)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, ServersResponse):
                raise ProtocolError(
                    f"Expected ServersResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            return response.nodes

    async def add(self, node_id: int, address: str) -> None:
        """Add a node to the cluster (Raft membership change). Must hit the leader.

        Lands the node as ``NodeRole.SPARE``; promote with :meth:`assign` after.
        """
        async with self._lock:
            request = AddRequest(node_id=node_id, address=address)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, EmptyResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

    async def assign(self, node_id: int, role: NodeRole) -> None:
        """Assign (or change) a node's role. Must be sent to the leader."""
        async with self._lock:
            request = AssignRequest(node_id=node_id, role=role)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, EmptyResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

    async def remove(self, node_id: int) -> None:
        """Remove a node from the cluster. Must be sent to the leader.

        Removing the current leader requires a prior :meth:`transfer`.
        """
        async with self._lock:
            request = RemoveRequest(node_id=node_id)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, EmptyResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

    async def describe(self) -> MetadataResponse:
        """Describe the connected node's metadata (failure_domain + weight).

        Describes the connected node, not the cluster; sweep via
        :meth:`ClusterClient.describe`.
        """
        async with self._lock:
            request = DescribeRequest(format=0)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, MetadataResponse):
                raise ProtocolError(
                    f"Expected MetadataResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            return response

    async def weight(self, weight: int) -> None:
        """Set the connected node's weight (leader-election preference).

        Affects only the connected node; sweep via
        :meth:`ClusterClient.set_weight`.
        """
        async with self._lock:
            request = WeightRequest(weight=weight)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, EmptyResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

    async def dump(self, database: str) -> dict[str, bytes]:
        """Dump a database to ``{filename: bytes}`` (DB file + WAL sidecar).

        The decode runs unconditionally on a worker thread (the payload is
        multi-MiB by design); the trailing-frame check stays on the loop.
        """
        async with self._lock:
            request = DumpRequest(name=database)
            await self._send_request(request)

            frame_bytes = await self._read_message_bytes()
            try:
                response = await asyncio.to_thread(
                    _decode_dump_response_sync, frame_bytes, self._decoder
                )
            except _WireProtocolError as e:
                raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {e}") from e

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, FilesResponse):
                raise ProtocolError(
                    f"Expected FilesResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            return response.files

    async def transfer(self, target_node_id: int) -> None:
        """Request leadership transfer to ``target_node_id``. Must hit the leader.

        Returns once the server accepts the request; election convergence is
        observable via a subsequent :meth:`get_leader`.
        """
        async with self._lock:
            request = TransferRequest(target_node_id=target_node_id)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, EmptyResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

    async def open_database(self, name: str, flags: int = 0, vfs: str = "") -> int:
        """Open a database; return its id.

        Upstream always assigns 0 to the first DB on a fresh connection, so any
        other id means a buggy/hostile peer and is rejected defensively.
        """
        async with self._lock:
            request = OpenRequest(name=name, flags=flags, vfs=vfs)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, DbResponse):
                raise ProtocolError(
                    f"Expected DbResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            if response.db_id != 0:
                # WIRE_DECODE_FAILED_PREFIX so SA's is_disconnect drops the slot.
                raise ProtocolError(
                    f"{WIRE_DECODE_FAILED_PREFIX}: OPEN returned db_id={response.db_id}, "
                    f"expected 0 (upstream contract: first DB on a fresh connection "
                    f"is always assigned id 0){self._addr_suffix()}"
                )

            return response.db_id

    async def prepare(self, db_id: int, sql: str) -> tuple[int, int]:
        """Prepare a SQL statement.

        Returns (stmt_id, num_params).
        """
        async with self._lock:
            request = PrepareRequest(db_id=db_id, sql=sql)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, StmtResponse):
                raise ProtocolError(
                    f"Expected StmtResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            # Confirm the echoed db_id matches; a mismatch means the server's
            # statement registry drifted, so invalidate rather than route
            # writes at the wrong DB. WIRE_DECODE_FAILED_PREFIX so SA's
            # is_disconnect drops the slot.
            if response.db_id != db_id:
                raise ProtocolError(
                    f"{WIRE_DECODE_FAILED_PREFIX}: StmtResponse db_id {response.db_id} "
                    f"does not match requested db_id {db_id}{self._addr_suffix()}"
                )

            return response.stmt_id, response.num_params

    async def finalize(self, db_id: int, stmt_id: int) -> None:
        """Finalize (close) a prepared statement."""
        async with self._lock:
            request = FinalizeRequest(db_id=db_id, stmt_id=stmt_id)
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, EmptyResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

    async def _interrupt(self, db_id: int) -> None:
        """Ask the server to stop producing rows for this db_id.

        Drains in-flight frames until the EmptyResponse ack arrives;
        FailureResponse mid-drain raises OperationalError. Currently unused
        in-tree (CancelledError invalidates the connection instead).
        """
        async with self._lock:
            request = InterruptRequest(db_id=db_id)
            await self._send_request(request)

            # Drain trailing frames until EMPTY, bounded by the operation
            # deadline and max_continuation_frames. Depending on timing the
            # gateway either replaces or appends a separate EMPTY ack, so the
            # wire may carry [RESULT-or-ROWS-stream, EMPTY]; treating RESULT as
            # terminal would leave the trailing EMPTY buffered and poison the
            # next RPC. Exit only on EMPTY (mirrors Go's Protocol.Interrupt).
            deadline = self._operation_deadline()
            frames = 0
            while True:
                # Check before the read so the cap N allows at most N frames.
                if (
                    self._max_continuation_frames is not None
                    and frames >= self._max_continuation_frames
                ):
                    raise ProtocolError(
                        f"Interrupt drain exceeded max_continuation_frames cap "
                        f"({self._max_continuation_frames}); server may be "
                        f"slow-dripping rows{self._addr_suffix()}."
                    )
                response = await self._read_response(deadline=deadline, allow_trailing=True)
                if isinstance(response, EmptyResponse):
                    return
                if isinstance(response, FailureResponse):
                    # FAILURE is terminal: it replaces the EMPTY ack on the
                    # interrupt-during-EXEC abort path (no FAILURE-then-EMPTY).
                    raise OperationalError(
                        self._failure_text(response), response.code, raw_message=response.message
                    )
                # ROWS / RESULT are drain-through; the EMPTY ack follows.
                # Any other type means stream desync.
                if not isinstance(response, (RowsResponse, ResultResponse)):
                    raise ProtocolError(
                        f"Expected EmptyResponse after Interrupt, got "
                        f"{type(response).__name__}{self._addr_suffix()}"
                    )
                # Guard the slow-frame DoS shape (RowsResponse-specific).
                if isinstance(response, RowsResponse) and not response.rows and response.has_more:
                    raise ProtocolError(
                        f"ROWS continuation made no progress during INTERRUPT "
                        f"drain: frame had 0 rows and has_more=True"
                        f"{self._addr_suffix()}"
                    )
                frames += 1
                # Yield: _read_response does not when frames are pre-buffered.
                await asyncio.sleep(0)

    async def exec_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[int, int]:
        """Execute SQL directly; return (last_insert_id, rows_affected).

        For multi-statement SQL, rows_affected is the last statement's count,
        NOT a sum across statements.
        """
        async with self._lock:
            request = ExecSqlRequest(
                db_id=db_id, sql=sql, params=params if params is not None else []
            )
            await self._send_request(request)

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

            if not isinstance(response, ResultResponse):
                raise ProtocolError(
                    f"Expected ResultResponse, got {type(response).__name__}{self._addr_suffix()}"
                )

            return response.last_insert_id, response.rows_affected

    async def _send_query(
        self, db_id: int, sql: str, params: Sequence[Any] | None
    ) -> tuple["RowsResponse", float]:
        """Send a QUERY_SQL request and return the first RowsResponse + deadline."""
        request = QuerySqlRequest(db_id=db_id, sql=sql, params=params if params is not None else [])
        await self._send_request(request)

        deadline = self._operation_deadline()
        response = await self._read_response(deadline=deadline)
        if isinstance(response, FailureResponse):
            raise OperationalError(
                self._failure_text(response), response.code, raw_message=response.message
            )
        if not isinstance(response, RowsResponse):
            raise ProtocolError(
                f"Expected RowsResponse, got {type(response).__name__}{self._addr_suffix()}"
            )
        return response, deadline

    async def _drain_continuations(
        self, initial: "RowsResponse", deadline: float
    ) -> tuple[list[list[Any]], list[list[int]]]:
        """Drain all continuation frames, enforcing progress + total-row caps.

        Returns ``(rows, row_types)`` with one type list per row (SQLite is
        dynamically typed, so a column's type can vary across rows).
        """
        # Check the cap against the initial frame too: a server can pack the
        # whole oversized result into one has_more=False frame.
        if self._max_total_rows is not None and len(initial.rows) > self._max_total_rows:
            raise ProtocolError(
                f"Query exceeded max_total_rows cap ({self._max_total_rows}); "
                f"reduce result size or raise the cap on the connection/pool."
            )
        # Alias the wire layer's fresh per-decode inner lists straight into the
        # accumulator (nothing else holds a reference), skipping a per-row copy.
        # ``cast`` widens ValueType (an IntEnum) to int per the return contract.
        all_rows: list[list[Any]] = list(initial.rows)
        all_row_types: list[list[int]] = cast("list[list[int]]", list(initial.row_types))
        response = initial
        frames = 1  # the initial frame counts
        while response.has_more:
            # Check before the read so the cap N allows at most N frames.
            if (
                self._max_continuation_frames is not None
                and frames >= self._max_continuation_frames
            ):
                raise ProtocolError(
                    f"Query exceeded max_continuation_frames cap "
                    f"({self._max_continuation_frames}); server may be "
                    f"slow-dripping rows."
                )
            next_response = await self._read_continuation(deadline=deadline)
            frames += 1
            if not next_response.rows and next_response.has_more:
                raise ProtocolError(
                    "ROWS continuation made no progress: frame had 0 rows and has_more=True"
                )
            if self._max_total_rows is not None and (
                len(all_rows) + len(next_response.rows) > self._max_total_rows
            ):
                raise ProtocolError(
                    f"Query exceeded max_total_rows cap ({self._max_total_rows}); "
                    f"reduce result size or raise the cap on the connection/pool."
                )
            all_rows.extend(next_response.rows)
            all_row_types.extend(cast("list[list[int]]", next_response.row_types))
            response = next_response
            # Yield: _read_continuation does not when frames are pre-buffered,
            # so a fast-burst server would otherwise pin the loop for the drain.
            await asyncio.sleep(0)
        return all_rows, all_row_types

    async def query_sql_typed(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """Execute a query; return (column_names, column_types, row_types, rows).

        ``column_types`` are first-frame wire ValueType tags; ``row_types`` has
        one list per row (SQLite is dynamically typed). A mid-stream failure
        raises before any rows are returned.
        """
        async with self._lock:
            response, deadline = await self._send_query(db_id, sql, params)
            column_names = list(response.column_names)
            column_types = [int(t) for t in response.column_types]
            all_rows, all_row_types = await self._drain_continuations(response, deadline)
            return column_names, column_types, all_row_types, all_rows

    async def query_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query directly; return (column_names, rows).

        Multi-statement SELECT is rejected server-side (no extra result sets
        to drain). Use :meth:`query_sql_typed` for per-column ValueType tags.
        """
        async with self._lock:
            response, deadline = await self._send_query(db_id, sql, params)
            column_names = response.column_names
            all_rows, _ = await self._drain_continuations(response, deadline)

            return column_names, all_rows

    async def _read_message_bytes(
        self,
        deadline: float | None = None,
        *,
        allow_trailing: bool = False,
    ) -> bytes:
        """Read the next complete frame as raw bytes, without decoding.

        Lets the dump path materialise off-loop. The trailing-frame check fires
        on the loop thread before any offload; ``allow_trailing`` opts out
        (the INTERRUPT drain expects two frames in sequence).
        """
        from dqlitewire.constants import ResponseType

        if deadline is None:
            deadline = self._operation_deadline()
        try:
            while not self._decoder.has_message():
                data = await self._read_data(deadline=deadline)
                self._decoder.feed(data)
                await asyncio.sleep(0)
            frame_bytes = self._decoder._buffer.read_message()
        except _WireProtocolError as e:
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {e}") from e
        if frame_bytes is None:  # pragma: no cover
            raise ProtocolError(f"Failed to read message bytes{self._addr_suffix()}")
        # msg_type lives at header offset 4. ROWS is the only type with
        # legitimate trailing frames; everything else is terminal.
        msg_type = frame_bytes[4]
        if not allow_trailing and msg_type != ResponseType.ROWS and self._decoder.has_message():
            raise ProtocolError(
                f"Server emitted extra response after message type "
                f"{msg_type}{self._addr_suffix()} — protocol "
                f"violation, invalidating connection"
            )
        return frame_bytes

    async def _send_request(self, request: Message) -> None:
        """Encode ``request`` and send via :meth:`_send`.

        Offloads the encode to a worker thread above
        ``_ENCODE_OFFLOAD_THRESHOLD`` so a multi-MiB payload does not freeze
        the loop. Cancel-safe because the encoder is stateless and the full
        frame is built before any byte is written; a streaming encoder would
        break both invariants and must invalidate on cancel.
        """
        if _estimate_request_body_size(request) >= _ENCODE_OFFLOAD_THRESHOLD:
            frame = await asyncio.to_thread(self._encoder.encode, request)
        else:
            frame = self._encoder.encode(request)
        await self._send(frame)

    async def _send(self, frame: bytes) -> None:
        """Write a frame and drain, wrapping transport errors.

        ``writer.write`` is inside the try because it can raise RuntimeError
        synchronously on a closing transport, which would otherwise leak past
        _run_protocol's except chain. The drain is bounded by self._timeout so
        a peer that stops reading cannot stall it indefinitely.
        """
        try:
            self._writer.write(frame)
            # asyncio.timeout (not wait_for) so an outer cancel does not discard
            # the drain's high-water-mark future. Mirrors the sibling sites.
            async with asyncio.timeout(self._timeout):
                await self._writer.drain()
        except TimeoutError as e:
            raise DqliteConnectionError(
                f"Write timeout{self._addr_suffix()} after {self._timeout}s"
            ) from e
        # ConnectionError et al are OSError subclasses (PEP 3151). RuntimeError
        # is kept separately for "Transport is closed" from writer.write.
        except (OSError, RuntimeError) as e:
            raise DqliteConnectionError(f"Write failed{self._addr_suffix()}: {e}") from e

    async def _read_data(self, deadline: float | None = None) -> bytes:
        """Read a chunk, bounded by a per-operation deadline.

        Capping the per-chunk timeout by the remaining budget stops a
        slow-drip server (just-under-timeout each chunk) keeping a call alive
        indefinitely. Transport errors wrap as DqliteConnectionError.
        """
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                # Report the actual overrun, not the per-read window.
                overrun = -remaining
                raise DqliteConnectionError(
                    f"Operation{self._addr_suffix()} exceeded deadline by {overrun:.3f}s"
                )
            # Clamp against clock drift between the guard and the call below.
            timeout = max(0.0, min(remaining, self._read_timeout))
        else:
            timeout = self._read_timeout
        try:
            # asyncio.timeout (not wait_for) so an outer cancel does not discard
            # bytes already buffered by the read — exactly-once for DML responses.
            async with asyncio.timeout(timeout):
                data = await self._reader.read(_READ_CHUNK_SIZE)
        except TimeoutError as e:
            raise DqliteConnectionError(
                f"Server read{self._addr_suffix()} timed out after {timeout:.1f}s"
            ) from e
        # See _send for OSError-subsumption rationale.
        except (OSError, RuntimeError) as e:
            raise DqliteConnectionError(f"Read failed{self._addr_suffix()}: {e}") from e
        if not data:
            raise DqliteConnectionError(f"Connection closed by server{self._addr_suffix()}")
        return data

    def _addr_suffix(self) -> str:
        """Render the peer address as a trailing ``" to <addr>"`` fragment.

        Empty when the address is unknown. Routes through the strict
        ``sanitize_for_log`` (escapes LF/TAB) because this text flows into log
        records; an LF-bearing address could otherwise splice rows (CWE-117).
        """
        if not self._address:
            return ""
        return f" to {_sanitize_for_log(self._address)}"

    def _failure_text(self, response: FailureResponse) -> str:
        """Render a FailureResponse as an exception body string.

        Truncates BEFORE composing the addr suffix so the suffix survives the
        display-message cap. Sanitises the server text (display variant) so a
        hostile peer cannot inject log-splitting characters.
        """
        from dqliteclient.cluster import _truncate_error

        truncated_msg = _truncate_error(response.message)
        safe_msg = _sanitize_display_text(truncated_msg)
        return _failure_message(safe_msg, self._addr_suffix())

    def _operation_deadline(self) -> float:
        """Deadline (monotonic seconds) for a single read-side operation.

        Uses ``_read_timeout`` (not ``_timeout``) so the
        ``trust_server_heartbeat`` widening flows through; the write path keeps
        ``_timeout``. The deadline is per-phase, not cumulative: an N-frame
        continuation can take N x timeout end-to-end (compounded by widening up
        to the 300 s cap). Wrap the outer call in ``asyncio.timeout`` for an
        absolute bound.
        """
        return asyncio.get_running_loop().time() + self._read_timeout

    async def _read_continuation(self, deadline: float | None = None) -> RowsResponse:
        """Read and decode a ROWS continuation frame.

        query_sql passes its own deadline so the budget spans every frame. An
        EmptyResponse here means an out-of-band server-side cancel (the client
        never sends INTERRUPT on this path); surface it as a protocol error.
        """
        if deadline is None:  # pragma: no cover
            # Defensive: in-tree callers always pass a deadline.
            deadline = self._operation_deadline()

        try:
            while True:
                # Threshold-gated decode offload (mirror of _read_response).
                # decode_continuation is stateful so it must run as a unit; the
                # _lock serialises decoder access.
                pending_size = self._compute_pending_size()
                if pending_size >= _DECODE_OFFLOAD_THRESHOLD:
                    result = await asyncio.to_thread(self._decoder.decode_continuation)
                else:
                    result = self._decoder.decode_continuation()
                if isinstance(result, EmptyResponse):
                    raise ProtocolError(
                        "Unexpected EmptyResponse during ROWS continuation"
                        f"{self._addr_suffix()}; query may have been "
                        "interrupted server-side."
                    )
                if result is not None:
                    return result
                data = await self._read_data(deadline=deadline)
                self._decoder.feed(data)
                # Yield: _read_data does not when the next chunk is pre-buffered,
                # so a fast-burst server would otherwise pin the loop.
                await asyncio.sleep(0)
        except _WireServerFailure as e:
            # Surface the SQLite code so SA/dbapi can classify. Pre-truncate
            # before the addr suffix so it survives the display-message cap.
            from dqliteclient.cluster import _truncate_error

            truncated_msg = _truncate_error(e.message)
            raise OperationalError(
                _failure_message(truncated_msg, self._addr_suffix()),
                e.code,
                raw_message=e.message,
            ) from e
        except _WireProtocolError as e:
            # Deliberately do NOT recover in-place via skip_message: wrap as a
            # client ProtocolError so _run_protocol invalidates the connection.
            # Hitting the 64 MiB cap means an attacker/buggy server/wire bug,
            # none safely resumable on the same socket — re-acquire instead.
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {e}") from e

    async def _read_response(
        self,
        deadline: float | None = None,
        *,
        allow_trailing: bool = False,
    ) -> Message:
        """Read and decode the next response message.

        Callers spanning multiple reads (query_sql) pass their own deadline so
        cumulative wall time is bounded. ``allow_trailing=True`` (INTERRUPT
        drain only) relaxes the no-extra-buffered-frames hardening below.
        """
        if deadline is None:
            deadline = self._operation_deadline()
        try:
            while not self._decoder.has_message():
                data = await self._read_data(deadline=deadline)
                self._decoder.feed(data)
                # Yield: defensive against a fast-burst server (see
                # _read_continuation).
                await asyncio.sleep(0)

            # Threshold-gated decode offload. ``decode`` is stateful (sets
            # continuation state, counters, column snapshots) so it must run as
            # a unit; the _lock makes the worker-thread call safe.
            pending_size = self._compute_pending_size()
            if pending_size >= _DECODE_OFFLOAD_THRESHOLD:
                message = await asyncio.to_thread(self._decoder.decode)
            else:
                message = self._decoder.decode()
        except _WireProtocolError as e:
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {e}") from e

        if message is None:  # pragma: no cover
            # Defensive: requires a torn buffer between has_message() and here.
            raise ProtocolError(f"Failed to decode message{self._addr_suffix()}")

        # Hostile-server hardening: every response but ROWS is terminal, so a
        # trailing buffered frame is extra/coalesced bytes — invalidate.
        # allow_trailing is the INTERRUPT drain's opt-out.
        if (
            not allow_trailing
            and not isinstance(message, RowsResponse)
            and self._decoder.has_message()
        ):
            raise ProtocolError(
                f"Server emitted extra response after "
                f"{type(message).__name__}{self._addr_suffix()} — protocol "
                "violation, invalidating connection"
            )

        return message

    def close(self) -> None:
        self._writer.close()

    async def wait_closed(self) -> None:
        await self._writer.wait_closed()
