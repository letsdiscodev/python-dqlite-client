"""Low-level protocol handler for dqlite."""

import asyncio
import logging
import secrets
import sys
from collections.abc import Sequence
from typing import Any, Final, NoReturn

from dqliteclient.exceptions import DqliteConnectionError, OperationalError, ProtocolError
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)
from dqlitewire import MessageDecoder, MessageEncoder
from dqlitewire.exceptions import (
    ProtocolError as _WireProtocolError,
)
from dqlitewire.exceptions import (
    ServerFailure as _WireServerFailure,
)
from dqlitewire.messages import (
    ClientRequest,
    DbResponse,
    EmptyResponse,
    ExecSqlRequest,
    FailureResponse,
    FinalizeRequest,
    InterruptRequest,
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

__all__ = ["DqliteProtocol"]

logger = logging.getLogger(__name__)

# Socket read buffer size. 4 KiB balances syscall overhead for typical
# request/response payloads against latency for small wire messages.
_READ_CHUNK_SIZE: Final[int] = 4096

# Upper bound on how wide a server's advertised heartbeat can stretch
# the per-read deadline on a connection that opted into
# ``trust_server_heartbeat``. Without this cap a hostile or buggy
# server could advertise an arbitrary value and effectively disable
# client-side read timeouts for the whole session. Sized to tolerate
# sane operational tuning (``config.c`` defaults to 15 s; 300 s fits
# 20× that with plenty of headroom) while still bounding the widening
# to a known scale. Changes here must be reflected in the
# ``trust_server_heartbeat`` docstrings on ``DqliteProtocol.__init__``
# and ``DqliteProtocol.handshake``, ``DqliteConnection.__init__``,
# and the top-level ``connect`` / ``create_pool`` docstrings in
# ``__init__.py``.
_HEARTBEAT_READ_TIMEOUT_CAP_SECONDS: Final[float] = 300.0


def _failure_message(message: str, addr_suffix: str) -> str:
    """Render the body of a FailureResponse-derived exception.

    Substitutes a stable placeholder when the server message is empty
    or reduces to whitespace under the wire-layer
    ``_sanitize_server_text`` cleanup. Without this, an empty message
    bubbles up as ``"[1] "`` (with a trailing space and no
    diagnostic), which log scraping cannot group on usefully and
    operators cannot grep. The placeholder ``"(no diagnostic from
    server)"`` is the contract.
    """
    body = message if message.strip() else "(no diagnostic from server)"
    return body + addr_suffix


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
        timeout: float = 10.0,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        address: str | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        # Forward the user's continuation caps so the codec's per-stream
        # counters honour the configured value. Without this, a user
        # explicitly bumping max_total_rows above the codec default
        # silently sees DecodeError at the codec cap; conversely, ``None``
        # ("disabled") at the protocol layer must NOT be forwarded as
        # ``None`` (the codec rejects ``None`` / ``< 1``). Translate
        # ``None`` to ``sys.maxsize`` so the codec never trips and the
        # client-layer (this class) remains the sole gatekeeper.
        decoder_max_total = max_total_rows if max_total_rows is not None else sys.maxsize
        decoder_max_frames = (
            max_continuation_frames if max_continuation_frames is not None else sys.maxsize
        )
        self._decoder = MessageDecoder(
            is_request=False,
            max_total_rows=decoder_max_total,
            max_continuation_frames=decoder_max_frames,
        )
        self._client_id = 0
        self._heartbeat_timeout = 0
        self._timeout = timeout
        # Per-read deadline, initially equal to the operator-configured
        # write/drain budget. When ``trust_server_heartbeat=True`` the
        # handshake may widen this (up to 300 s) based on the server's
        # advertised heartbeat; the write-path ``self._timeout`` never
        # changes so a hostile server cannot stretch the operator's
        # write SLO by advertising a long heartbeat.
        self._read_timeout = timeout
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

    def __reduce__(self) -> NoReturn:
        # Wraps a live ``asyncio.StreamReader`` / ``StreamWriter``
        # (loop-bound), a ``MessageDecoder`` with internal buffer
        # state, and per-stream cap counters that mean nothing
        # post-deserialise. Surface a clear driver-level TypeError
        # instead of leaking the underlying ``cannot pickle
        # 'asyncio.streams.StreamReader'``.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — wraps "
            f"loop-bound StreamReader / StreamWriter and a stateful "
            f"MessageDecoder; reconstruct from a fresh wire handshake "
            f"in the target process instead."
        )

    @property
    def is_wire_coherent(self) -> bool:
        """True if the decoder buffer has not been poisoned.

        A poisoned buffer (mid-stream desync, malformed frame, etc.)
        cannot recover without ``reset()`` + reconnect. Pool reset
        consults this before sending ROLLBACK so a wasted round-trip
        on an already-doomed connection is short-circuited.

        INTENTIONALLY consulted only by the pool reset path
        (``pool.py::_socket_looks_dead``). The dbapi, the SA dialect,
        and direct ``DqliteConnection`` callers route a wire desync
        through the exception-based classifier chain instead:
        wire-layer ``ProtocolError`` is wrapped to
        ``OperationalError(code=None)`` by
        ``dqlitedbapi.cursor._call_client``, and the SA dialect's
        ``is_disconnect`` substring branch matches the
        ``"wire decode failed"`` prefix the client emits. That
        layering is deliberate — wire coherence is a hint, not a
        liveness contract: a ``CancelledError`` mid-flight could
        poison the buffer between this read and the next operation,
        so a ``True`` return MUST NOT be treated as "the connection
        is healthy". Use as a short-circuit optimisation only; do
        not propagate this check to ``do_ping`` / pre-checkout
        paths or to other operational classifiers.
        """
        return not self._decoder.is_poisoned

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
            # Mirror the query-path raise sites: surface the
            # server-reported code and peer address so log aggregators
            # can group on the numeric code (DQLITE_PARSE,
            # DQLITE_NOTLEADER, etc.) rather than on text alone.
            raise ProtocolError(
                f"Handshake failed: [{response.code}] {self._failure_text(response)}"
            )

        if not isinstance(response, WelcomeResponse):
            raise ProtocolError(
                f"Expected WelcomeResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

        self._client_id = client_id
        self._heartbeat_timeout = response.heartbeat_timeout
        # Use the server-advertised heartbeat only when explicitly
        # trusted. Previously we always widened ``self._timeout`` up
        # to 300 s based on the server value, which let a hostile
        # server amplify the operator's configured timeout up to 30×.
        # Default is opt-in (``trust_server_heartbeat=False``): the
        # per-read-deadline widening is DISABLED unless the caller
        # explicitly enables it. The server-advertised value is still
        # read here for diagnostics but has no effect on the deadline.
        if self._trust_server_heartbeat and response.heartbeat_timeout > 0:
            heartbeat_seconds = response.heartbeat_timeout / 1000.0
            # Cap to prevent a malicious/buggy server from disabling timeouts.
            # Only widen the READ deadline — the write-path self._timeout
            # stays pinned to the operator-configured value so a hostile
            # server cannot advertise a long heartbeat to stretch every
            # writer.drain() beyond the operator's SLO.
            new_read_timeout = max(
                self._read_timeout, min(heartbeat_seconds, _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS)
            )
            if new_read_timeout != self._read_timeout:
                # Security-relevant opt-in: surface the actual widening
                # at DEBUG so an operator who flipped the knob can
                # confirm it took effect.
                logger.debug(
                    "handshake: widened per-read timeout %.2fs -> %.2fs (server heartbeat=%.2fs)",
                    self._read_timeout,
                    new_read_timeout,
                    heartbeat_seconds,
                )
                self._read_timeout = new_read_timeout
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
            raise OperationalError(response.code, self._failure_text(response))

        if not isinstance(response, LeaderResponse):
            raise ProtocolError(
                f"Expected LeaderResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

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
            raise OperationalError(response.code, self._failure_text(response))

        if not isinstance(response, DbResponse):
            raise ProtocolError(
                f"Expected DbResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

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
            raise OperationalError(response.code, self._failure_text(response))

        if not isinstance(response, StmtResponse):
            raise ProtocolError(
                f"Expected StmtResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

        # Defense-in-depth: confirm the server's StmtResponse echoes
        # the db_id we asked it to prepare against. A mismatch would
        # mean the server's prepared-statement registry has drifted
        # from the client's view, and any future exec/finalize
        # against the returned stmt_id would target a different
        # database. Surface this as a ProtocolError so the
        # connection is invalidated rather than silently routing
        # writes against the wrong DB.
        if response.db_id != db_id:
            # Prefix with the canonical "wire decode failed" phrase so
            # SA's ``is_disconnect`` substring matcher routes this
            # through the pool-invalidate path. Without the prefix,
            # the registry-drift event would surface as a non-
            # disconnect ProtocolError and the SA pool would keep the
            # broken slot. The prefix matches the wire-decode
            # invalidation already wired into
            # ``sqlalchemy-dqlite._dqlite_disconnect_messages``.
            raise ProtocolError(
                f"wire decode failed: StmtResponse db_id {response.db_id} "
                f"does not match requested db_id {db_id}{self._addr_suffix()}"
            )

        return response.stmt_id, response.num_params

    async def finalize(self, db_id: int, stmt_id: int) -> None:
        """Finalize (close) a prepared statement."""
        request = FinalizeRequest(db_id=db_id, stmt_id=stmt_id)
        self._writer.write(request.encode())
        await self._send()

        response = await self._read_response()

        if isinstance(response, FailureResponse):
            raise OperationalError(response.code, self._failure_text(response))

        if not isinstance(response, EmptyResponse):
            raise ProtocolError(
                f"Expected EmptyResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

    async def interrupt(self, db_id: int) -> None:
        """Ask the server to stop producing further rows for this db_id.

        Drains the stream by consuming messages until an
        ``EmptyResponse`` arrives. The server may have continuation
        ``RowsResponse`` frames in flight at the moment we call this;
        the drain loop swallows them and returns when the final
        ``EmptyResponse`` acknowledges the interrupt.

        ``FailureResponse`` mid-drain is raised as
        ``OperationalError`` — the interrupt itself may have been
        refused. Other unexpected message types are ``ProtocolError``.

        .. note::

            **Currently unused by in-tree code paths.** On
            ``asyncio.CancelledError`` mid-query, ``DqliteConnection``
            invalidates the local connection but does NOT send
            INTERRUPT — the server continues the query until its own
            completion path, which on large result sets can amplify
            cluster resource use. This mirrors the Go / C clients'
            ``Rows.Close`` / ``clientSendInterrupt`` paths in wire
            shape only; wiring them into cursor-cancel / task-cancel
            is a future-streaming-support feature that is deliberately
            out of scope for the current synchronous-drain client.

        .. warning::

            ``interrupt`` has no internal concurrency guard at the
            protocol layer — it writes to the shared ``_writer`` and
            reads from the shared decoder without any lock. The
            ``DqliteConnection.execute`` / ``query_sql`` callers at
            the level above hold ``_in_use``, so two callers there
            cannot interleave. But ``interrupt`` itself is reachable
            via the private ``conn._protocol`` path. External callers
            using it MUST follow this exact ordering:

            1. cancel the outer task that owns the in-flight stmt,
            2. **await** the cancelled task to completion so the
               reader half has fully exited (a bare ``cancel()`` does
               not block — the cancelled task may still have a
               pending ``_read_response`` queued),
            3. only then invoke ``interrupt(db_id)``.

            Skipping (2) races the cancelled task's read against
            ``interrupt``'s read on the shared decoder and produces
            wire desync.
        """
        request = InterruptRequest(db_id=db_id)
        self._writer.write(request.encode())
        await self._send()

        # Drain: swallow any trailing continuation frames, break when
        # EmptyResponse arrives. Bound by the single operation deadline
        # so a non-responsive server cannot stall this forever, and by
        # the max_continuation_frames cap so a slow-dripping server
        # cannot pin the client on per-frame decode work inside that
        # deadline window (same rationale as _drain_continuations).
        deadline = self._operation_deadline()
        frames = 0
        while True:
            response = await self._read_response(deadline=deadline)
            if isinstance(response, EmptyResponse):
                return
            # ``ResultResponse`` is the EXEC-side terminal: when the
            # interrupted statement was an EXEC / EXEC_SQL whose
            # done-callback emitted RESULT before the INTERRUPT
            # took effect, the response queue holds the RESULT
            # before the EmptyResponse acknowledgement. Treat it as
            # equivalent to EmptyResponse — the wire is in a
            # coherent state and the interrupt has been honoured.
            # Without this branch, a cancel landing on an EXEC
            # path would hit the "Expected EmptyResponse" arm and
            # poison the wire.
            if isinstance(response, ResultResponse):
                return
            if isinstance(response, FailureResponse):
                raise OperationalError(response.code, self._failure_text(response))
            # RowsResponse mid-drain is expected: the server's in-flight
            # continuation may land before the interrupt takes effect.
            # Other message types indicate stream desync.
            if not isinstance(response, RowsResponse):
                raise ProtocolError(
                    f"Expected EmptyResponse after Interrupt, got "
                    f"{type(response).__name__}{self._addr_suffix()}"
                )
            # No-progress check: a server emitting empty rows frames with
            # has_more=True before EmptyResponse would consume up to
            # ``max_continuation_frames`` iterations of decode work,
            # mirroring the slow-frame DoS shape the query path
            # already defends against. Mirror the discipline.
            if not response.rows and response.has_more:
                raise ProtocolError(
                    f"ROWS continuation made no progress during INTERRUPT "
                    f"drain: frame had 0 rows and has_more=True"
                    f"{self._addr_suffix()}"
                )
            frames += 1
            if self._max_continuation_frames is not None and frames > self._max_continuation_frames:
                raise ProtocolError(
                    f"Interrupt drain exceeded max_continuation_frames cap "
                    f"({self._max_continuation_frames}); server may be "
                    f"slow-dripping rows{self._addr_suffix()}."
                )
            # Fall through: any RowsResponse (has_more=True or False)
            # means more frames may still arrive before the
            # terminating EmptyResponse.

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
            raise OperationalError(response.code, self._failure_text(response))

        if not isinstance(response, ResultResponse):
            raise ProtocolError(
                f"Expected ResultResponse, got {type(response).__name__}{self._addr_suffix()}"
            )

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
            raise OperationalError(response.code, self._failure_text(response))
        if not isinstance(response, RowsResponse):
            raise ProtocolError(
                f"Expected RowsResponse, got {type(response).__name__}{self._addr_suffix()}"
            )
        return response, deadline

    async def _drain_continuations(
        self, initial: "RowsResponse", deadline: float
    ) -> tuple[list[list[Any]], list[list[int]]]:
        """Drain all continuation frames, enforcing the progress + total-row caps.

        Returns ``(rows, row_types)`` — a flat list of all row values
        (initial frame first) and a parallel list of per-row wire
        ``ValueType`` tags. SQLite is dynamically typed, so row types
        can vary per row; callers that apply result-side converters
        need the per-row list rather than a collapsed first-row view.

        A continuation claiming more rows with zero delivered, or a
        cumulative row count exceeding ``max_total_rows``, raises
        ProtocolError.
        """
        # Fire the cumulative-total cap against the initial frame first.
        # A hostile or buggy server can pack the whole oversized result
        # into a single frame with has_more=False and never enter the
        # continuation loop; without this pre-check the governor would
        # only apply to the continuation tail.
        if self._max_total_rows is not None and len(initial.rows) > self._max_total_rows:
            raise ProtocolError(
                f"Query exceeded max_total_rows cap ({self._max_total_rows}); "
                f"reduce result size or raise the cap on the connection/pool."
            )
        all_rows = list(initial.rows)
        all_row_types: list[list[int]] = [[int(t) for t in rt] for rt in initial.row_types]
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
            all_row_types.extend([int(t) for t in rt] for rt in next_response.row_types)
            response = next_response
        return all_rows, all_row_types

    async def query_sql_typed(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """Execute a query and return (column_names, column_types, row_types, rows).

        ``column_types`` are the wire-level ``ValueType`` integer tags
        from the first response frame — what DBAPI cursor.description
        maps into ``type_code``. ``row_types`` carries one list per
        row so callers can apply result-side converters per-row (SQLite
        is dynamically typed; two rows in the same column can carry
        different wire types under UNION, ``CASE``, ``COALESCE``, or
        ``typeof()``).

        Atomicity: a mid-stream server failure or an unexpected message
        type raises before any rows are returned to the caller; the
        local row list is discarded. The connection is invalidated so
        callers don't accidentally reuse it with torn protocol state.
        """
        response, deadline = await self._send_query(db_id, sql, params)
        column_names = list(response.column_names)
        column_types = [int(t) for t in response.column_types]
        all_rows, all_row_types = await self._drain_continuations(response, deadline)
        return column_names, column_types, all_row_types, all_rows

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
        all_rows, _ = await self._drain_continuations(response, deadline)

        return column_names, all_rows

    async def _send(self) -> None:
        """Drain the writer, wrapping transport errors as DqliteConnectionError.

        A peer that accepts the TCP connection but stops reading can stall
        ``drain()`` indefinitely on the high-water-mark future. Bound the
        drain by ``self._timeout`` so the caller-configured timeout is
        authoritative for sends just as it is for reads.
        """
        try:
            await asyncio.wait_for(self._writer.drain(), timeout=self._timeout)
        except TimeoutError as e:
            raise DqliteConnectionError(
                f"Write timeout{self._addr_suffix()} after {self._timeout}s"
            ) from e
        # ConnectionError / BrokenPipeError / ConnectionResetError /
        # ConnectionAbortedError / ConnectionRefusedError are all OSError
        # subclasses since PEP 3151 — the bare OSError arm already catches
        # them. See sqlalchemy-dqlite/src/sqlalchemydqlite/base.py:362-368
        # for the project's source-of-truth on this idiom. RuntimeError is
        # kept (not an OSError subclass) to cover "Transport is closed".
        except (OSError, RuntimeError) as e:
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
                    f"Operation{self._addr_suffix()} exceeded {self._read_timeout}s deadline"
                )
            # Clamp against drift: the loop clock can advance between
            # the guard above and the ``wait_for`` call below. Without
            # ``max(0.0, ...)`` a sub-microsecond drift could hand
            # ``wait_for`` a negative timeout. 3.13 handles negative
            # timeouts consistently (immediate TimeoutError) but the
            # clamp documents intent.
            timeout = max(0.0, min(remaining, self._read_timeout))
        else:
            timeout = self._read_timeout
        try:
            data = await asyncio.wait_for(self._reader.read(_READ_CHUNK_SIZE), timeout=timeout)
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

        Returns an empty string when the address is unknown — keeping
        error messages clean for callers that don't thread it in.
        """
        return f" to {self._address}" if self._address else ""

    def _failure_text(self, response: FailureResponse) -> str:
        """Render a FailureResponse as the body string for an
        OperationalError / ProtocolError raise.

        Substitutes a stable placeholder when the server message is
        empty or whitespace-only so log scraping has a keyword to
        match instead of staring at ``"[1] "``. Wraps
        :func:`_failure_message` with the protocol's ``_addr_suffix``.
        Used uniformly across the query-path raise sites so the
        rendering is consistent.
        """
        return _failure_message(response.message, self._addr_suffix())

    def _operation_deadline(self) -> float:
        """Deadline (monotonic seconds) for a single protocol operation.

        Note on multi-phase RPCs: ``timeout`` bounds each phase
        independently — a phase-1 ``_send`` followed by a phase-2
        ``_read_response`` followed by a continuation drain can
        cumulatively take up to ``timeout`` per phase. Worst-case
        wall-clock for a single ``query_sql`` is therefore on the
        order of ``2 × timeout`` (send + read+drain), and a fresh
        ``connect → query`` flow can stack ``handshake`` +
        ``open_database`` + ``query_sql`` for proportionally more.
        Callers needing an absolute end-to-end bound should wrap the
        outer call in ``asyncio.timeout`` / ``asyncio.wait_for``.
        This matches go-dqlite's per-phase budgeting.
        """
        return asyncio.get_running_loop().time() + self._timeout

    async def _read_continuation(self, deadline: float | None = None) -> RowsResponse:
        """Read and decode a ROWS continuation frame.

        If ``deadline`` is None, a fresh per-operation deadline is set;
        query_sql passes its own deadline so the budget spans every
        continuation frame, not each one individually.

        ``decode_continuation`` may also return an ``EmptyResponse`` —
        the server's acknowledgement of a mid-stream INTERRUPT. The
        client-layer ``query_sql`` flow does not send INTERRUPT, so an
        ``EmptyResponse`` here would mean the server-side query was
        cancelled out-of-band; surface it as a protocol error.
        """
        if deadline is None:  # pragma: no cover
            # Defensive: in-tree callers always pass an explicit
            # deadline. Reaching here requires a third-party caller
            # using ``read_continuation`` directly; the fallback
            # picks up the connection's operation timeout.
            deadline = self._operation_deadline()
        try:
            while True:
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
        except _WireServerFailure as e:
            # Server-authored failure mid-stream: surface the SQLite code
            # so sqlalchemy's is_disconnect and dbapi's code-to-exception
            # map can classify correctly (leader flip, constraint, etc.).
            raise OperationalError(e.code, e.message) from e
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

        if message is None:  # pragma: no cover
            # Defensive: ``decoder.decode()`` returns None only when
            # the buffer has no complete message. ``read_message``
            # is called only after ``has_message()`` confirmed a
            # complete frame is available, so this branch requires
            # a torn buffer state between the two calls — verified
            # by code review, not coverage.
            raise ProtocolError(f"Failed to decode message{self._addr_suffix()}")

        # Hostile-server hardening: a ``FailureResponse`` is always
        # terminal per the dqlite wire spec — one request, one
        # response. If the decoder still has another frame buffered
        # after extracting a FailureResponse, the server emitted
        # extra bytes (or two coalesced replies arrived in one TCP
        # segment). Without this check the leftover frame would be
        # consumed as the response to the NEXT user request,
        # producing a misleading ``OperationalError`` against an
        # unrelated operation. Raise ``ProtocolError`` here so
        # ``_run_protocol`` invalidates the connection and the pool
        # discards the slot.
        if isinstance(message, FailureResponse) and self._decoder.has_message():
            raise ProtocolError(
                f"Server emitted extra response after FailureResponse"
                f"{self._addr_suffix()} — protocol violation, invalidating connection"
            )

        return message

    def close(self) -> None:
        """Close the connection."""
        self._writer.close()

    async def wait_closed(self) -> None:
        """Wait for the connection to close."""
        await self._writer.wait_closed()
