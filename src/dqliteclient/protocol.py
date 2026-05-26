"""Low-level protocol handler for dqlite.

Also hosts the public ``validate_positive_int_or_none`` validator
because that helper originated inline inside
:class:`DqliteProtocol`'s ``max_total_rows`` /
``max_continuation_frames`` kwargs and was promoted in place. The
sibling public validators ``validate_timeout`` and ``parse_address``
live in :mod:`dqliteclient.connection` for the same first-caller
reason. All three are re-exported via :mod:`dqliteclient`; the
asymmetric module homes are an artefact of where each validator was
first needed, not a contract.
"""

import asyncio
import logging
import secrets
from collections.abc import Sequence
from typing import Any, Final, NoReturn

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

# Default cap on inbound message-frame size, re-exported from the wire
# layer's ``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE`` (64 MiB). Promoted to
# a module-level constant so the propagation through DqliteProtocol /
# DqliteConnection / ConnectionPool sees a single source of truth, in
# parity with ``_DEFAULT_MAX_TOTAL_ROWS`` / ``_DEFAULT_MAX_CONTINUATION_FRAMES``.
# A wire-layer bump propagates automatically; a client-side hot-fix
# can rebind this constant without touching the wire.
DEFAULT_MAX_MESSAGE_SIZE: Final[int] = ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE


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


def validate_positive_int_or_none(value: int | None, name: str) -> int | None:
    """Shared validation for positive-int-or-None parameters.

    Used for both ``max_total_rows`` and ``max_continuation_frames``.
    None disables the cap; any int value must be > 0.

    Public so downstream packages (``dqlitedbapi``, ``sqlalchemy-dqlite``)
    can apply the same construction-time validation without reaching
    into private symbols. Same shape as the public ``parse_address`` /
    ``allowlist_policy`` helpers in this package.
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
        max_message_size: int | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        # Forward the user's continuation caps directly. The codec
        # accepts ``None`` to disable a cap (its public contract);
        # passing through verbatim keeps the client-layer (this
        # class) and the wire-layer cap surfaces aligned without an
        # encoded sentinel. ``max_message_size`` is the third wire
        # governor (alongside max_total_rows / max_continuation_frames);
        # None falls back to the wire-layer default (64 MiB) — see
        # ``DEFAULT_MAX_MESSAGE_SIZE`` for the re-export rationale.
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
        # Symmetric outbound cap. ``MessageEncoder.encode()`` rejects
        # frames whose total size exceeds ``max_message_size`` BEFORE
        # they reach the writer, so an accidentally oversized
        # ``PrepareRequest`` (huge SQL string) or ``ExecSqlRequest``
        # (oversized bind value) surfaces as a local ``EncodeError``
        # instead of a remote rejection after a round-trip or a
        # stalled write pipe. Sharing the same cap value with the
        # decoder keeps the lever single-knob from the operator's
        # perspective. Go-dqlite has no outbound cap; this is purely
        # Python defence-in-depth.
        self._encoder = MessageEncoder(max_message_size=effective_max_message_size)
        self._max_message_size = effective_max_message_size
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
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        # Per-query frame cap. Complements max_total_rows: a server
        # sending 10M 1-row frames to reach the row cap would still
        # burn 10M × decode-cost of Python work; the frame cap bounds
        # that at ~100k iterations.
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        # When True, the client honors the server-advertised heartbeat
        # timeout to adjust its per-read deadline (subject to the 300 s
        # hard cap). When False (default), the server value is recorded
        # for diagnostics only and the operator-configured ``timeout``
        # is authoritative. Opt-in protects operators whose timeout is
        # a latency-SLO boundary from server-induced amplification.
        self._trust_server_heartbeat = trust_server_heartbeat
        # Serialise wire-touching RPCs. The dqlite server does not
        # support concurrent requests on a single connection — two
        # concurrent ``Call``s on the same protocol interleave their
        # writes on the shared writer and their reads on the shared
        # decoder, surfacing as a malformed-frame ``ProtocolError`` or
        # codec poisoning several round-trips later with no breadcrumb
        # pointing at the concurrency violation. Mirrors go-dqlite's
        # ``Protocol.mu sync.Mutex`` at ``internal/protocol/protocol.go:15-30``
        # ("We need to take a lock since the dqlite server currently
        # does not support concurrent requests.").
        #
        # In-tree callers (``DqliteConnection.execute`` /
        # ``query_sql`` / admin) hold ``_in_use`` one layer up which
        # guards the same race; the protocol-layer lock closes the gap
        # for third-party callers that import ``DqliteProtocol``
        # directly and share an instance across tasks. ``_send`` /
        # ``_read_*`` are called from inside locked methods so do not
        # need their own acquisition.
        #
        # ``handshake`` / ``negotiate_protocol_only`` are connect-time
        # methods called from a single coroutine before the protocol
        # is published anywhere; they intentionally skip the lock.
        self._lock = asyncio.Lock()

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
        """True if the codec-layer decoder buffer has not been poisoned.

        Reflects ONLY the wire-codec layer (``MessageDecoder.is_poisoned``).
        A poisoned buffer (mid-stream wire desync, malformed frame, etc.)
        cannot recover without ``reset()`` + reconnect; the pool-reset
        path consults this before sending ROLLBACK so a wasted round-trip
        on an already-doomed connection is short-circuited.

        Does NOT reflect client-layer ``ProtocolError`` raises that
        detect higher-level invariant violations on top of coherent
        bytes: ``prepare`` db_id mismatch, ``_read_response``
        extra-frame-after-FAILURE, ``_read_continuation`` unexpected
        EmptyResponse, ``interrupt`` drain wrong-type / no-progress /
        frame-cap, ``_drain_continuations`` max_total_rows /
        max_continuation_frames / no-progress. In those cases the codec
        decoded correctly but the connection still has unread frames
        buffered or a continuation expected; the wire IS desynchronised
        at the next-request boundary, but ``is_wire_coherent`` returns
        ``True``.

        This narrowness is intentional. ``is_wire_coherent`` is the
        codec-poison hint only; client-layer protocol violations route
        through ``DqliteConnection._run_protocol``'s
        ``ProtocolError → _invalidate`` chain, which closes the writer
        and clears ``self._protocol`` — the pool short-circuits on
        ``protocol is None`` before consulting this accessor at all
        (see ``pool.py::_socket_looks_dead``).

        INTENTIONALLY consulted only by the pool reset path. The
        dbapi, the SA dialect, and direct ``DqliteConnection``
        callers route a wire desync through the exception-based
        classifier chain instead: wire-layer ``ProtocolError`` is
        wrapped to ``OperationalError(code=None)`` by
        ``dqlitedbapi.cursor._call_client``, and the SA dialect's
        ``is_disconnect`` substring branch matches the
        ``"wire decode failed"`` prefix the client emits.

        Wire coherence is a hint, not a liveness contract: a
        ``CancelledError`` mid-flight could poison the buffer between
        this read and the next operation, so a ``True`` return MUST
        NOT be treated as "the connection is healthy". Use as a
        short-circuit optimisation only; do not propagate this check
        to ``do_ping`` / pre-checkout paths or to other operational
        classifiers. Third-party harnesses that consume
        ``DqliteProtocol`` directly should always invalidate on any
        ``ProtocolError`` rather than reading this flag.
        """
        return not self._decoder.is_poisoned

    async def negotiate_protocol_only(self) -> None:
        """Probe-only handshake: write the protocol version bytes
        WITHOUT a follow-up ``ClientRequest``.

        The C server reads 8 bytes (the version) and validates them
        BEFORE expecting the next message; no response is sent. The
        server's request handlers (notably ``handle_leader``) do not
        require ``g->client_id`` to be set, so a leader probe can skip
        registration. ``cluster._query_leader`` uses this lighter path
        so probes against non-leader peers don't allocate a per-client
        server slot.

        DO NOT use this for connections that issue real queries —
        ``handle_open`` / ``handle_exec`` etc. depend on a registered
        client in some code paths and on per-connection state set up
        by ``handle_client``. Use :meth:`handshake` for those paths.
        """
        await self._send(self._encoder.encode_handshake())

    async def handshake(self, client_id: int | None = None) -> int:
        """Perform protocol handshake.

        If ``client_id`` is not provided, a random non-zero 63-bit id is
        generated so each connection is distinguishable in server logs,
        traces, and per-client metrics. Returns the heartbeat timeout
        from the server.

        Bundles the protocol-version exchange and the ``ClientRequest``
        registration into a SINGLE writer-write so the kernel can pack
        them into one TCP segment. Splitting into two writes would
        double the syscall count and the per-segment overhead on the
        connect hot path. The probe path (leader discovery) uses the
        lighter :meth:`negotiate_protocol_only` instead.
        """
        if client_id is None:
            # Deliberate divergence from go-dqlite. Go declares
            # ``Connector.clientID uint64`` (``connector.go:75-83``)
            # default-zero and never assigns; every Go connection
            # registers with ``id=0``. The server-side
            # ``handle_client`` (``gateway.c:300-309``) stores
            # ``g->client_id = request.id`` verbatim — used only
            # for ``tracef`` records and per-connection
            # ``handle_interrupt`` keying, neither of which requires
            # a unique id per connection.
            #
            # Python randomises so each connection is distinguishable
            # in server logs, traces, and per-client metrics — useful
            # for operators tailing a busy server and filtering by
            # connection id. 63 bits avoids sign-extension pitfalls
            # if an intermediate layer treats the id as int64. The
            # ``or 1`` guards against the astronomically unlikely
            # all-zero draw — preserved deliberately so an
            # operator's ``client_id != 0`` filter is reliable.
            #
            # Operational trade-offs:
            #
            # * Mixed-client clusters: an operator filtering server
            #   logs by ``client_id`` sees Python connections with
            #   random 63-bit ids while Go connections all show id 0;
            #   the same filter cannot apply uniformly across both
            #   client implementations.
            # * No cross-reconnect correlation: a Python client that
            #   loses connection and reconnects gets a FRESH random
            #   id, so server-side per-client metrics see one
            #   connection per attempt rather than one client with
            #   several attempts. Go's id=0 doesn't solve this either,
            #   but the current Python design forecloses a future
            #   operator-supplied stable id without an API change.
            # * ``secrets.randbits`` (CSPRNG-backed via
            #   ``getrandom(2)`` on Linux) is the choice over the
            #   faster ``random.getrandbits`` because the
            #   ``random._inst`` PRNG state is process-global and
            #   not fork-aware (see ``retry._retry_random`` and
            #   ``cluster._cluster_random`` for the same reasoning
            #   at sibling sites). The per-handshake cost is
            #   sub-microsecond and negligible against handshake
            #   RTT; the fork-safety property is the load-bearing
            #   reason for this choice.
            client_id = secrets.randbits(63) or 1
        # Send protocol version + client registration together
        request = ClientRequest(client_id=client_id)
        # Record the negotiated id BEFORE the wire write so the
        # ``FailureResponse`` arm below has the slot-allocation
        # breadcrumb available. Upstream ``handle_client``
        # (``gateway.c:300-309``) writes ``g->client_id =
        # request.id`` BEFORE composing the response, so by the time
        # any FailureResponse reaches us the server-side per-gateway
        # slot has already been allocated. Reclamation happens at
        # TCP close (the caller's ``_abort_protocol`` drives this
        # via ``writer.close() + wait_closed``); surfacing the id in
        # the exception message means an operator triaging a
        # handshake failure can correlate the server-side trace
        # without walking back to ``gateway.c``.
        self._client_id = client_id
        await self._send(self._encoder.encode_handshake() + self._encoder.encode(request))

        # Read welcome response
        response = await self._read_response()

        if isinstance(response, FailureResponse):
            # Mirror the query-path raise sites: surface the
            # server-reported code and peer address so log aggregators
            # can group on the numeric code (DQLITE_PARSE,
            # DQLITE_NOTLEADER, etc.) rather than on text alone.
            # ``_failure_text`` already pre-truncates the body before
            # appending the addr suffix, bounding the result at
            # ~240 chars; do NOT wrap it in an outer ``_truncate_error``,
            # which would re-truncate over the addr suffix and silently
            # strip the peer attribution exactly when the operator
            # needs it most (a long handshake failure).
            # Preserve the verbatim server text via ``raw_message`` —
            # mirrors the 16 sibling FailureResponse-derived raise
            # sites in this file. Without this, ``ProtocolError.raw_message``
            # defaults to the synthetic "Handshake failed: ..."
            # string, throwing away the verbatim peer text (capped at
            # ``DqliteError._MAX_RAW_MESSAGE`` codepoints, default 4 KiB)
            # that ``ProtocolError.raw_message`` is meant to carry for
            # cross-process forensic recovery.
            # Raise OperationalError (with the structured ``code``
            # second positional) rather than ProtocolError so the
            # server-supplied SQLite code is preserved as an
            # attribute, not only interpolated into the message
            # string. Mirrors the 16 sibling FailureResponse-derived
            # raise sites; the prior ProtocolError site was the lone
            # asymmetric path that forced downstream classifiers
            # (``_connect_impl``'s leader-flip arm, dbapi's
            # ``_CODE_TO_EXCEPTION``, SA's ``is_disconnect``) to
            # fall back on substring matching. Semantically the
            # handshake failure carries a server-supplied SQLite
            # code — operational, not protocol-shape.
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
        # Surface the three diagnostic edge cases the wire layer accepts
        # but cannot remediate from the client side. Each is non-fatal
        # (the protocol stays operational) but the operator chasing a
        # mis-configured-peer / non-conforming-server symptom needs
        # the breadcrumb to correlate against per-cluster config audits.
        if response.heartbeat_timeout == 0:
            # Wire-layer ``WelcomeResponse.heartbeat_timeout`` docstring
            # flags 0 as semantically ambiguous: upstream ``config.c``
            # defaults to 15000 and never emits 0, so 0 from the wire is
            # either a misconfigured peer or a non-conforming server.
            logger.debug(
                "handshake: server advertised heartbeat=0 (semantically "
                "ambiguous per wire spec; widening disabled)"
            )
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
            if heartbeat_seconds > _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS:
                # Cap firing: surface at WARNING so an operator can
                # tell that the server-advertised value was over-large
                # and was clipped at the client cap.
                logger.warning(
                    "handshake: server-advertised heartbeat %.2fs exceeds "
                    "client cap %.2fs; clipping",
                    heartbeat_seconds,
                    _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS,
                )
                heartbeat_seconds = _HEARTBEAT_READ_TIMEOUT_CAP_SECONDS
            # Cap to prevent a malicious/buggy server from disabling timeouts.
            # Only widen the READ deadline — the write-path self._timeout
            # stays pinned to the operator-configured value so a hostile
            # server cannot advertise a long heartbeat to stretch every
            # writer.drain() beyond the operator's SLO.
            new_read_timeout = max(self._read_timeout, heartbeat_seconds)
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
            else:
                # No-op widening: the server's value was smaller than
                # the operator's configured deadline. The operator
                # opted into ``trust_server_heartbeat`` expecting
                # widening; surface the mismatch so they can
                # recalibrate either the server config or the client
                # read deadline.
                logger.debug(
                    "handshake: trust_server_heartbeat=True but server "
                    "advertised %.2fs <= configured read timeout %.2fs; "
                    "no widening applied",
                    heartbeat_seconds,
                    self._read_timeout,
                )
        return response.heartbeat_timeout

    async def get_leader(self) -> tuple[int, str]:
        """Request leader information.

        Returns ``(node_id, address)``. The ``(0, "")`` "no leader
        known" shape is passed through verbatim — callers normalise it
        upstream — but the malformed ``(0, nonempty)`` shape is
        rejected here as a ``ProtocolError``: upstream
        ``raft_leader`` pairs id and address (both filled, or both
        zero/NULL) so ``(0, addr)`` is either a confused or hostile
        peer. Guarding at the wire layer is symmetric with sibling
        RPC defences (e.g. ``prepare``'s ``db_id`` mismatch) and
        means third-party callers of ``DqliteProtocol`` get the same
        defence as the cluster-layer wrappers.

        The mirror shape ``(N, "")`` is NOT raised here. The cluster
        wrappers want to log the ``RAFT_NOMEM`` transient with the
        per-address context that the protocol layer lacks before
        normalising to "no leader known"; moving the raise up would
        lose that breadcrumb.
        """
        async with self._lock:
            request = LeaderRequest()
            await self._send(self._encoder.encode(request))

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
                # ``raft_leader`` never emits ``node_id=0`` paired
                # with a non-empty address; a peer returning this is
                # either confused or hostile. Reject at the wire
                # layer so any consumer of LEADER replies — including
                # third-party callers wiring ``DqliteProtocol``
                # directly into custom probes — gets the defence.
                raise ProtocolError(
                    f"server returned address "
                    f"{_sanitize_display_text(response.address)!r} "
                    f"with node_id=0; expected both or neither"
                    f"{self._addr_suffix()}"
                )

            return response.node_id, response.address

    async def cluster(self) -> list[NodeInfo]:
        """Request the cluster's node list.

        Sends ``ClusterRequest(format=1)`` and returns the V1 node
        list (id + address + role) the server replies with via
        :class:`ServersResponse`. Format 0 (V0, no role field) is
        rejected client-side by ``ClusterRequest.__post_init__``;
        callers needing the V0 shape would have to bypass this layer.

        Mirrors the spec-level admin operation ``go-dqlite/client.Cluster``.
        Any node can answer this — the cluster view is replicated —
        but the typical caller asks the leader for the freshest view.
        """
        async with self._lock:
            request = ClusterRequest(format=1)
            await self._send(self._encoder.encode(request))

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
        """Add a node to the cluster (Raft membership change).

        Sends ``AddRequest(node_id, address)`` and expects
        :class:`EmptyResponse`. The peer MUST be the current leader;
        a follower returns ``SQLITE_IOERR_NOT_LEADER``-style codes.

        Mirrors go-dqlite's ``client.go::EncodeAdd`` half of
        ``Client.Add``. Per upstream semantics, ADD lands the node as
        ``NodeRole.SPARE``; promote with :meth:`assign` after.
        """
        async with self._lock:
            request = AddRequest(node_id=node_id, address=address)
            await self._send(self._encoder.encode(request))

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
        """Assign (or change) a node's role.

        Sends ``AssignRequest(node_id, role)`` (modern 16-byte body;
        the legacy 8-byte PROMOTE shape is encoder-rejected per the
        wire-layer documentation). Expects :class:`EmptyResponse`.
        Must be sent to the leader.

        Mirrors go-dqlite's ``client.go::EncodeAssign``.
        """
        async with self._lock:
            request = AssignRequest(node_id=node_id, role=role)
            await self._send(self._encoder.encode(request))

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
        """Remove a node from the cluster (Raft membership change).

        Sends ``RemoveRequest(node_id)`` and expects
        :class:`EmptyResponse`. Must be sent to the leader. Removing
        the current leader requires a prior :meth:`transfer` to a
        different voter — the server otherwise rejects.

        Mirrors go-dqlite's ``client.go::EncodeRemove``.
        """
        async with self._lock:
            request = RemoveRequest(node_id=node_id)
            await self._send(self._encoder.encode(request))

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
        """Describe the connected node's metadata.

        Sends ``DescribeRequest(format=0)`` (the only format the
        upstream gateway accepts) and returns the
        :class:`MetadataResponse` carrying ``failure_domain`` and
        ``weight``. The response describes the **connected node**,
        not the cluster — to sweep, the higher-level
        :meth:`ClusterClient.describe` accepts an explicit
        ``address``.

        Mirrors go-dqlite's ``client.go::EncodeDescribe``.
        """
        async with self._lock:
            request = DescribeRequest(format=0)
            await self._send(self._encoder.encode(request))

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
        """Set the connected node's weight.

        Sends ``WeightRequest(weight)`` and expects
        :class:`EmptyResponse`. Weight tunes leader-election
        preference within a failure domain. Affects only the
        **connected node** — to sweep, the higher-level
        :meth:`ClusterClient.set_weight` accepts an explicit
        ``address``.

        Mirrors go-dqlite's ``client.go::EncodeWeight``.
        """
        async with self._lock:
            request = WeightRequest(weight=weight)
            await self._send(self._encoder.encode(request))

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
        """Dump a database to ``{filename: bytes}``.

        Sends ``DumpRequest(database)`` and returns the
        :class:`FilesResponse`'s ``files`` dict (typically two
        entries: the database file and its WAL sidecar). The
        wire-layer enforces caps on file count + per-file size +
        8-byte content alignment so a hostile peer cannot exhaust
        client memory.

        Mirrors go-dqlite's ``client.go::EncodeDump``.
        """
        async with self._lock:
            request = DumpRequest(name=database)
            await self._send(self._encoder.encode(request))

            response = await self._read_response()

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
        """Request leadership transfer to ``target_node_id``.

        Sends ``TransferRequest(target_node_id)`` and expects an
        :class:`EmptyResponse` on success. The peer connected to MUST
        be the current leader; sending Transfer to a follower returns
        ``SQLITE_IOERR_NOT_LEADER`` (or equivalent), which surfaces
        here as :class:`OperationalError` so the higher-level
        ``ClusterClient.transfer_leadership`` can rediscover the
        leader and retry against it.

        On success, Raft begins promoting ``target_node_id`` to
        leader; the call returns once the server has accepted the
        transfer request. Election convergence (the new leader being
        able to accept writes) is observable via a subsequent
        :meth:`get_leader` call.

        Mirrors the spec-level admin operation ``go-dqlite/client.Transfer``.
        """
        async with self._lock:
            request = TransferRequest(target_node_id=target_node_id)
            await self._send(self._encoder.encode(request))

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
        """Open a database.

        Returns the database ID. Upstream contractually assigns ``0``
        to the first (and only) database opened on a fresh connection
        — ``gateway.c::handle_open`` writes ``response.id = 0`` and
        the next OPEN on the same connection is refused with
        ``SQLITE_BUSY`` (``gateway.c:319-324``). A wire response that
        echoes any other id is either a buggy / misconfigured server
        or a hostile peer; reject defensively rather than threading
        the bad id through every subsequent RPC and surfacing the
        symptom one round-trip later via ``prepare``'s ``db_id``
        mismatch guard. Mirrors that guard's discipline.

        Uses ``WIRE_DECODE_FAILED_PREFIX`` so SA's ``is_disconnect``
        triggers pool invalidation downstream — the connection's
        view of the database id is irrecoverable on this RPC, so the
        pool must drop the slot.
        """
        async with self._lock:
            request = OpenRequest(name=name, flags=flags, vfs=vfs)
            await self._send(self._encoder.encode(request))

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
            await self._send(self._encoder.encode(request))

            response = await self._read_response()

            if isinstance(response, FailureResponse):
                raise OperationalError(
                    self._failure_text(response), response.code, raw_message=response.message
                )

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
                # Prefix with the canonical ``WIRE_DECODE_FAILED_PREFIX``
                # phrase so SA's ``is_disconnect`` substring matcher routes
                # this through the pool-invalidate path. Without the
                # prefix, the registry-drift event would surface as a
                # non-disconnect ProtocolError and the SA pool would keep
                # the broken slot. The prefix matches the wire-decode
                # invalidation wired into
                # ``sqlalchemy-dqlite._dqlite_disconnect_messages``.
                raise ProtocolError(
                    f"{WIRE_DECODE_FAILED_PREFIX}: StmtResponse db_id {response.db_id} "
                    f"does not match requested db_id {db_id}{self._addr_suffix()}"
                )

            return response.stmt_id, response.num_params

    async def finalize(self, db_id: int, stmt_id: int) -> None:
        """Finalize (close) a prepared statement."""
        async with self._lock:
            request = FinalizeRequest(db_id=db_id, stmt_id=stmt_id)
            await self._send(self._encoder.encode(request))

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

        .. note::

            ``interrupt`` is serialised behind the protocol-layer
            ``self._lock`` together with all other wire-touching RPCs.
            An ``interrupt`` issued while another task holds the lock
            (e.g. is in the middle of ``query_sql``) queues until that
            RPC completes, which is the correct ordering — the
            in-flight read drains its own response before
            ``interrupt`` writes its INTERRUPT frame. Callers no
            longer need to cancel-and-await the in-flight task
            manually before invoking ``interrupt``; the lock enforces
            the ordering automatically. (For abortive cancellation
            mid-RPC, ``DqliteConnection._invalidate`` is the
            higher-level escape hatch: it tears down the protocol
            without waiting for in-flight reads.)
        """
        async with self._lock:
            request = InterruptRequest(db_id=db_id)
            await self._send(self._encoder.encode(request))

            # Drain: swallow any trailing continuation frames, break when
            # EmptyResponse arrives. Bound by the single operation deadline
            # so a non-responsive server cannot stall this forever, and by
            # the max_continuation_frames cap so a slow-dripping server
            # cannot pin the client on per-frame decode work inside that
            # deadline window (same rationale as _drain_continuations).
            #
            # Loop-till-EmptyResponse mirrors Go's ``Protocol.Interrupt``
            # (``/tmp/go-dqlite/internal/protocol/protocol.go:103-113``).
            # The gateway dispatches INTERRUPT one of two ways
            # (``dqlite-upstream/src/gateway.c:1366-1390``):
            #
            # 1. If a request is still in flight (``g->req != NULL``) when
            #    the INTERRUPT lands, the dispatcher calls ``interrupt(g)``
            #    (sets ``cancellation_requested`` / aborts ``leader_exec``)
            #    and the in-flight RPC's done-callback emits the response
            #    — RESULT or FAILURE for EXEC, EMPTY for cancelled QUERY,
            #    plus any in-flight ROWS continuation frames already queued.
            #    No separate EMPTY for the INTERRUPT itself in this path.
            # 2. If the in-flight RPC's done-callback has ALREADY fired
            #    (``g->req = NULL`` per ``gateway.c:536`` for EXEC,
            #    ``:780`` for QUERY) before the INTERRUPT is dispatched,
            #    the dispatcher falls through to ``handle_interrupt``
            #    (``gateway.c:952-960``) which DOES emit a separate EMPTY
            #    for the INTERRUPT ack. Wire then carries
            #    ``[RESULT-or-ROWS-stream, EMPTY]`` — the prior RPC's
            #    response followed by the INTERRUPT's own EMPTY.
            #
            # Treating RESULT as the terminal would consume case (2)'s
            # prior-RPC frame and leave the trailing EMPTY in the decoder
            # buffer, poisoning the next RPC on the connection. Drain
            # through RESULT (and ROWS) and only exit on EMPTY — same
            # discipline as Go.
            deadline = self._operation_deadline()
            frames = 0
            while True:
                # Cap is checked BEFORE the read so the documented bound
                # ``N`` allows AT MOST ``N`` non-EMPTY frames (a check
                # after the read would fire AFTER reading the (N+1)-th).
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
                    # FAILURE is terminal: the C server replaces the EMPTY
                    # ack with FAILURE on the interrupt-during-EXEC abort
                    # path (``handle_exec_done_cb`` calls ``exec_failure``
                    # at ``gateway.c:542-544`` when ``raft_status != 0``).
                    # There is no FAILURE-then-EMPTY pair because the
                    # failure callback nulls ``g->req`` before the
                    # dispatcher could read the next request.
                    raise OperationalError(
                        self._failure_text(response), response.code, raw_message=response.message
                    )
                # ROWS and RESULT are drain-through frames in the
                # interrupt context. ROWS is the in-flight QUERY's
                # continuation stream; RESULT is the in-flight EXEC's
                # terminal — either way, the EMPTY for the INTERRUPT ack
                # is the next frame and we keep reading. Other message
                # types indicate stream desync.
                if not isinstance(response, (RowsResponse, ResultResponse)):
                    raise ProtocolError(
                        f"Expected EmptyResponse after Interrupt, got "
                        f"{type(response).__name__}{self._addr_suffix()}"
                    )
                # No-progress check is RowsResponse-specific: a server
                # emitting empty rows frames with has_more=True before
                # EmptyResponse would consume up to
                # ``max_continuation_frames`` iterations of decode work,
                # mirroring the slow-frame DoS shape the query path
                # already defends against. ResultResponse has no
                # ``has_more`` field — RESULT is a single frame per
                # in-flight RPC and the cap below counts it.
                if isinstance(response, RowsResponse) and not response.rows and response.has_more:
                    raise ProtocolError(
                        f"ROWS continuation made no progress during INTERRUPT "
                        f"drain: frame had 0 rows and has_more=True"
                        f"{self._addr_suffix()}"
                    )
                frames += 1
                # Fall through: keep reading until EmptyResponse arrives.

    async def exec_sql(
        self, db_id: int, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[int, int]:
        """Execute SQL directly.

        Returns (last_insert_id, rows_affected). For multi-statement SQL
        (semicolon-separated), the server aggregates internally and returns
        a single RESULT with sqlite3_changes() of the last statement only —
        rows_affected is NOT a sum across statements.
        """
        async with self._lock:
            request = ExecSqlRequest(
                db_id=db_id, sql=sql, params=params if params is not None else []
            )
            await self._send(self._encoder.encode(request))

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
        """Send a QUERY_SQL request and return the first RowsResponse + deadline.

        Raises OperationalError for server FailureResponse and ProtocolError
        for any other unexpected message type.
        """
        request = QuerySqlRequest(db_id=db_id, sql=sql, params=params if params is not None else [])
        await self._send(self._encoder.encode(request))

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
        # ``ValueType`` is an ``IntEnum`` (subclass of ``int``); the
        # wire layer already returns IntEnum members and downstream
        # consumers do dict lookups by int value, so a per-cell
        # ``int(t)`` is a runtime no-op that allocates a fresh ``int``
        # per cell. A shallow ``list(rt)`` skips the per-cell cost.
        # The outer list still needs copying because the wire layer's
        # row-type lists are not safe to alias into our cumulative
        # buffer.
        all_row_types: list[list[int]] = [list(rt) for rt in initial.row_types]
        response = initial
        frames = 1  # the initial frame counts
        while response.has_more:
            # Per-frame cap complements max_total_rows: a
            # slow-drip server sending 1-row-per-frame would
            # otherwise pin a client CPU with O(n) iterations of
            # decode work, where n is max_total_rows. Check BEFORE
            # the read so the documented cap ``N`` allows AT MOST
            # ``N`` decoded frames (initial + continuations counted
            # uniformly); a check after the read fires AFTER reading
            # the (N+1)-th frame even when ``has_more=False`` would
            # have ended the loop on the next iteration anyway.
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
            all_row_types.extend(list(rt) for rt in next_response.row_types)
            response = next_response
            # Cooperative loop yield. ``await self._read_continuation``
            # above does NOT yield to the loop scheduler when the next
            # frame is already buffered in the StreamReader — asyncio's
            # ``read*`` returns synchronously on that fast path. Without
            # this explicit yield, a fast-burst server that prefetches
            # many small frames pins the loop for the entire drain,
            # starving sibling coroutines (heartbeat probes, pool
            # acquirers, SA do_ping keepalives). The per-iteration
            # ``sleep(0)`` cost is sub-microsecond and dominated by the
            # per-frame decode work upstream.
            await asyncio.sleep(0)
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
        async with self._lock:
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
        by the server with OperationalError(message, code=SQLITE_ERROR) where message is "nonempty
        statement tail") — there are no additional result sets to drain.
        Use :meth:`query_sql_typed` to also get per-column ``ValueType``
        tags.
        """
        async with self._lock:
            response, deadline = await self._send_query(db_id, sql, params)
            column_names = response.column_names
            all_rows, _ = await self._drain_continuations(response, deadline)

            return column_names, all_rows

    async def _send(self, frame: bytes) -> None:
        """Write a frame and drain, wrapping transport errors as DqliteConnectionError.

        The synchronous ``self._writer.write(frame)`` is part of the
        protected scope: CPython's ``_SelectorSocketTransport.write``
        raises ``RuntimeError("Connection lost")`` /
        ``RuntimeError("Transport is closed")`` synchronously when the
        underlying transport is in the closing state (e.g. peer RST
        between two requests on the same pooled connection, concurrent
        ``_invalidate`` from a fork-pid mismatch). Without the write
        inside the try/except, that RuntimeError leaked past
        ``_run_protocol``'s except chain — which catches
        DqliteConnectionError / ProtocolError / OperationalError /
        cancel-class / _WireEncodeError but NOT bare RuntimeError —
        bypassing the invalidate arm and leaving the connection in a
        live-but-dead state.

        A peer that accepts the TCP connection but stops reading can
        stall ``drain()`` indefinitely on the high-water-mark future.
        Bound the drain by ``self._timeout`` so the caller-configured
        timeout is authoritative for sends just as it is for reads.
        """
        try:
            self._writer.write(frame)
            # Use ``asyncio.timeout`` cancel-scope semantics rather
            # than ``asyncio.wait_for`` so an outer cancel landing
            # while ``drain()`` is in flight does not discard the
            # writer's high-water-mark future. Mirrors the discipline
            # at the sibling dial / connect / admin / close sites
            # (``cluster.py::_query_leader``, ``cluster.py::
            # open_admin_connection``, ``connection.py::_connect_impl``,
            # ``connection.py::_close_impl``,
            # ``connection.py::_abort_protocol``,
            # ``connection.py::_invalidate``).
            # ``drain()`` returns ``None`` so the result-discard concern
            # does not apply to ``_send`` today — the migration is
            # sibling-parity defence so a future refactor that adds a
            # result-carrying drain does not silently regress.
            async with asyncio.timeout(self._timeout):
                await self._writer.drain()
        except TimeoutError as e:
            raise DqliteConnectionError(
                f"Write timeout{self._addr_suffix()} after {self._timeout}s"
            ) from e
        # ConnectionError / BrokenPipeError / ConnectionResetError /
        # ConnectionAbortedError / ConnectionRefusedError are all OSError
        # subclasses since PEP 3151 — the bare OSError arm already catches
        # them. See the canonical OSError / PEP-3151 idiom in
        # ``sqlalchemydqlite.base`` (the dialect's ``is_disconnect``
        # OSError-cause-walk). RuntimeError is
        # kept (not an OSError subclass) to cover "Transport is closed"
        # raised synchronously from ``writer.write``.
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
                # Report the actual budget that was overrun, not the
                # per-read window. With ``trust_server_heartbeat=True``
                # the per-read ``self._read_timeout`` may be widened
                # well above the configured ``self._timeout``; with
                # heartbeat widening disabled the deadline budget is
                # ``self._timeout``. Either way, ``-remaining`` is the
                # observed overrun and is what operators want to see.
                overrun = -remaining
                raise DqliteConnectionError(
                    f"Operation{self._addr_suffix()} exceeded deadline by {overrun:.3f}s"
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
            # Use ``asyncio.timeout`` cancel-scope semantics rather
            # than ``asyncio.wait_for`` so an outer cancel landing
            # while ``reader.read()`` is in flight does not discard
            # the bytes the inner future has already buffered (the
            # load-bearing concern: at-most-once vs exactly-once for
            # non-idempotent DML responses streaming back). Mirrors
            # the discipline at the sibling dial / connect / admin
            # sites; see ``_send`` for the sibling-parity rationale.
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

        Returns an empty string when the address is unknown — keeping
        error messages clean for callers that don't thread it in.

        Routes the address through ``sanitize_for_log`` (the strict
        variant that ESCAPES LF/TAB and strips U+2028/U+2029/bidi
        controls), NOT ``sanitize_server_text`` (display variant
        that preserves LF for multi-line readability). The exception
        text built here flows into downstream
        ``logger.error("%s", exc)`` / ``logger.exception(...)``
        sites; a peer that supplied an LF-bearing address — either
        via leader-redirect, a malformed node store entry, or a
        dial_func override that bypassed ``parse_address`` — would
        otherwise produce log records that splice across rows
        (CWE-117). Mirrors the discipline applied to
        ``ClusterClient._ProbeMiss.message``.
        """
        if not self._address:
            return ""
        return f" to {_sanitize_for_log(self._address)}"

    def _failure_text(self, response: FailureResponse) -> str:
        """Render a FailureResponse as the body string for an
        OperationalError / ProtocolError raise.

        Substitutes a stable placeholder when the server message is
        empty or whitespace-only so log scraping has a keyword to
        match instead of staring at ``"[1] "``. Wraps
        :func:`_failure_message` with the protocol's ``_addr_suffix``.
        Used uniformly across the query-path raise sites so the
        rendering is consistent.

        Truncates the server message BEFORE composing the addr
        suffix so the suffix survives the
        ``OperationalError._MAX_DISPLAY_MESSAGE`` codepoint cap on
        the exception's display ``message`` field. Without
        pre-truncation, a long server message (e.g. ORM-generated
        SQL with many bound parameters) would push the suffix past
        the cutoff — operators tailing logs via
        ``logger.error("%s", exc)`` would see the truncated server
        text but lose the peer-address attribution.

        Routes the server text through ``sanitize_server_text``
        (display variant — preserves LF / Tab for multi-line server
        diagnostics, strips control / bidi / invisible codepoints)
        so a hostile peer cannot inject log-splitting characters
        into the exception's display message. The leader-flip
        rewrap arm in ``connection.py`` already applies the same
        helper; mirror it on the canonical query-path raise.
        """
        from dqliteclient.cluster import _truncate_error

        truncated_msg = _truncate_error(response.message)
        safe_msg = _sanitize_display_text(truncated_msg)
        return _failure_message(safe_msg, self._addr_suffix())

    def _operation_deadline(self) -> float:
        """Deadline (monotonic seconds) for a single read-side protocol
        operation.

        Uses ``self._read_timeout`` (NOT ``self._timeout``) so the
        ``trust_server_heartbeat`` widening — which raises
        ``_read_timeout`` up to the server-advertised heartbeat
        capped at ``_HEARTBEAT_READ_TIMEOUT_CAP_SECONDS`` — actually
        flows through to the per-read deadline budget. Without this,
        the widening was dead code: ``_read_data``'s
        ``min(remaining, self._read_timeout)`` always selected
        ``remaining`` because the deadline came from
        ``self._timeout`` (un-widened), so the user-configured
        opt-in had no effect.

        Write-path ``_send`` continues to use ``self._timeout``
        directly — that's intentional: only the read
        side gets the heartbeat widening.

        Note on multi-phase RPCs: ``timeout`` bounds each phase
        independently — a phase-1 ``_send`` followed by a phase-2
        ``_read_response`` followed by a continuation drain can
        cumulatively take up to ``timeout`` per phase. Worst-case
        wall-clock for a single ``query_sql`` is therefore on the
        order of ``2 × timeout`` (send + read+drain), and a fresh
        ``connect → query`` flow can stack ``handshake`` +
        ``open_database`` + ``query_sql`` for proportionally more.
        Concrete worst-case quantification: a continuation-paginated
        result with N frames pays N × ``timeout`` because each
        continuation frame consumes its own per-read deadline (no
        cumulative budget bounds the read sequence). With
        ``timeout=10s`` and a 10-frame continuation, end-to-end
        wall-clock can reach ~120 s (send + read + 10 × continuation
        read) while every individual phase stayed within the
        operator's configured budget. Callers needing an absolute
        end-to-end bound should wrap the outer call in
        ``asyncio.timeout`` / ``asyncio.wait_for``.

        Server-heartbeat amplification: when
        ``trust_server_heartbeat=True`` widens ``_read_timeout`` up
        to the 300 s hard cap, the PER-PHASE budget grows; the
        end-to-end multiplier compounds. A 5-phase RPC with the
        server advertising the cap can stack 5 × 300 s = 1500 s of
        read time even when the operator set ``timeout=10``. The
        cap is per-phase, not cumulative — opting into the widening
        is opting out of the operator's per-RPC SLO at the
        cumulative level. Operators whose ``timeout`` is a
        latency-SLO boundary should NOT opt in (the default is
        ``False``).

        Note: this differs from go-dqlite, which uses an absolute
        per-call deadline (``conn.SetDeadline(ctx.Deadline())`` at
        ``connector.go:312-333`` — applied to BOTH the
        ``conn.Write`` and any subsequent ``conn.Read`` until
        cleared). The per-phase shape here is a deliberate
        Python-side choice — the asyncio.wait_for wraps individual
        sends and reads independently — and is documented at
        :class:`DqliteConnection` for caller guidance. The
        :class:`DqliteConnection` and :class:`ConnectionPool`
        docstrings echo the worst-case multiplier so operators
        sizing ``timeout`` for SLO-bound queries see the same shape
        at every entry point.
        """
        return asyncio.get_running_loop().time() + self._read_timeout

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
            # Pre-truncate the body BEFORE composing the addr suffix so
            # the suffix survives the OperationalError._MAX_DISPLAY_MESSAGE
            # codepoint cap on the exception's display field — matching
            # the eight sibling ``raise OperationalError`` sites in this
            # module that route through ``self._failure_text``.
            from dqliteclient.cluster import _truncate_error

            truncated_msg = _truncate_error(e.message)
            raise OperationalError(
                _failure_message(truncated_msg, self._addr_suffix()),
                e.code,
                raw_message=e.message,
            ) from e
        except _WireProtocolError as e:
            # Wire-decode failures are NOT recovered via
            # ``MessageDecoder.skip_message``. The wire layer documents
            # oversize-message rejection as recoverable in-place
            # (``read_message`` raises ``DecodeError`` without poisoning
            # so a caller can drain the over-large frame via
            # ``skip_message`` + ``feed`` and resume on the same
            # connection). The client layer deliberately opts out: we
            # wrap every wire-level ProtocolError into a client-level
            # ProtocolError, which ``_run_protocol`` routes through
            # ``_invalidate`` to drop the connection. Rationale: a
            # 64 MiB cap is high enough that hitting it almost always
            # indicates an attacker, a misbehaving server, or a
            # deeply-nested wire bug — none of which the client can
            # safely resume from on the same socket. The pool's
            # re-acquire path (handshake + OPEN) is the right
            # recovery, not in-place skip. The ``skip_message`` API
            # remains available for third-party harnesses that
            # consume ``DqliteProtocol`` directly and want a
            # different policy.
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {e}") from e

    async def _read_response(
        self,
        deadline: float | None = None,
        *,
        allow_trailing: bool = False,
    ) -> Message:
        """Read and decode the next response message.

        If ``deadline`` is None, a fresh per-operation deadline is set for
        this one response; callers that span multiple reads (e.g. query_sql
        across continuation frames) pass an externally-held deadline so
        the cumulative wall time is bounded.

        ``allow_trailing=True`` relaxes the "no extra buffered frames"
        check below. The INTERRUPT drain loop sets this because it
        deliberately consumes multiple frames in sequence (the prior
        RPC's terminal response followed by the EMPTY ack for the
        INTERRUPT itself) — see ``_interrupt`` for the gateway.c race
        that produces RESULT-then-EMPTY pairs. Every other caller leaves
        the default ``False`` so the hostile-server hardening below
        still fires on the one-request-one-response wire contract.
        """
        if deadline is None:
            deadline = self._operation_deadline()
        try:
            while not self._decoder.has_message():
                data = await self._read_data(deadline=deadline)
                self._decoder.feed(data)

            message = self._decoder.decode()
        except _WireProtocolError as e:
            raise ProtocolError(f"{WIRE_DECODE_FAILED_PREFIX}{self._addr_suffix()}: {e}") from e

        if message is None:  # pragma: no cover
            # Defensive: ``decoder.decode()`` returns None only when
            # the buffer has no complete message. ``read_message``
            # is called only after ``has_message()`` confirmed a
            # complete frame is available, so this branch requires
            # a torn buffer state between the two calls — verified
            # by code review, not coverage.
            raise ProtocolError(f"Failed to decode message{self._addr_suffix()}")

        # Hostile-server hardening: every dqlite response other than
        # ``RowsResponse`` is terminal per the wire spec — one request,
        # one response. If the decoder still has another frame
        # buffered after extracting a non-RowsResponse, the server
        # emitted extra bytes (or two coalesced replies arrived in
        # one TCP segment). Without this check the leftover frame
        # would be consumed as the response to the NEXT user request,
        # producing a misleading ``OperationalError`` against an
        # unrelated operation. Raise ``ProtocolError`` here so
        # ``_run_protocol`` invalidates the connection and the pool
        # discards the slot.
        #
        # ``RowsResponse`` is the sole exception: continuation frames
        # for a single Rows result legitimately carry across multiple
        # decode steps. The continuation drain path does its own
        # has_message-aware iteration.
        #
        # ``allow_trailing=True`` is the INTERRUPT-drain caller's
        # explicit opt-out (see docstring): the drain context
        # legitimately expects multiple frames in sequence (RESULT
        # from the just-completed EXEC, then EMPTY for the INTERRUPT
        # ack — gateway.c re-dispatches via ``handle_interrupt`` when
        # ``g->req == NULL`` at the time the INTERRUPT lands).
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
        """Close the connection."""
        self._writer.close()

    async def wait_closed(self) -> None:
        """Wait for the connection to close."""
        await self._writer.wait_closed()
