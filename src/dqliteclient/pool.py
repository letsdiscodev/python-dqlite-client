"""Connection pooling for dqlite."""

import asyncio
import contextlib
import logging
import math
import os
import warnings
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Final, NoReturn, Self

from dqliteclient import connection as _conn_mod
from dqliteclient._dial import DialFunc
from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import (
    _TRANSACTION_ROLLBACK_SQL,
    CLOSE_TIMEOUT_FLOOR_RATIONALE,
    DqliteConnection,
    _is_no_tx_rollback_error,
    get_current_pid,
    validate_timeout,
)
from dqliteclient.exceptions import (
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import NodeStore
from dqliteclient.protocol import validate_positive_int_or_none
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)
from dqlitewire import LEADER_ERROR_CODES, sanitize_for_log

__all__ = ["ConnectionPool"]

# Exception categories a best-effort pool cleanup (ROLLBACK, close())
# can legitimately raise on a partially-torn-down transport. Anything
# outside this tuple — ``CancelledError``, ``KeyboardInterrupt``,
# ``SystemExit``, and programming errors (``AttributeError``,
# ``TypeError``, ``RuntimeError``) — must propagate so structured
# concurrency and refactor bugs are observable. Mirrors the narrowing
# in ``AsyncAdaptedConnection.close`` and ``DqliteConnection.transaction``.
# ``OSError`` subsumes ``TimeoutError`` / ``BrokenPipeError`` /
# ``ConnectionError`` / ``ConnectionResetError``. A single
# ``OSError`` entry covers every stdlib transport-error shape —
# mirrors the classification in ``sqlalchemy-dqlite``'s
# ``is_disconnect``.
#
# ``InterfaceError`` belongs here defensively: ``_reset_connection``
# calls ``conn.execute("ROLLBACK")``, which goes through
# ``_run_protocol → _check_in_use``. A connection where ``_in_transaction``
# is ``True`` and ``_tx_owner`` is some other still-live task makes
# ``_check_in_use`` raise ``InterfaceError("owned by another task")``.
# That should drop the connection, not crash the pool's release path
# (which would leak a ``_size`` slot and eventually wedge the pool).
_POOL_CLEANUP_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    OSError,
    DqliteConnectionError,
    ProtocolError,
    OperationalError,
    InterfaceError,
)

logger = logging.getLogger(__name__)


def _socket_looks_dead(conn: DqliteConnection) -> bool:
    """Best-effort local detection of a half-closed TCP socket.

    Returns True if any of:

    * ``_protocol`` is missing or None — the connection was never
      handshaked, or has been invalidated; we cannot peek a transport,
      so we treat as dead so the caller drops it.
    * ``_protocol.is_wire_coherent`` is False — wire desync from a
      prior parse failure; the next round-trip is wasted RTT.
    * ``transport.is_closing()`` is True — peer FIN already observed.
    * ``reader.at_eof()`` is True — same shape, observed via reader.

    Returns False otherwise. Mocked / partially-built objects whose
    transport peek attributes (``transport``, ``reader``) are missing
    or raise ``AttributeError`` / ``RuntimeError`` are treated as
    alive for those individual peeks — the function only flags an
    affirmative dead signal from the transport / reader paths.

    The "missing ``_protocol`` is dead" branch is intentional: a real
    connection always has the attribute (possibly None for never-
    connected / invalidated state); the only way to reach this branch
    in production is a connection that genuinely has no usable
    protocol.

    Loop-affinity invariant: this function is only ever called on the
    loop's owning thread (acquire-path pre-ping is sync within
    ``ConnectionPool.acquire``'s coroutine; never crosses an
    ``await`` between the attribute peeks). A concurrent
    ``DqliteConnection._invalidate`` runs on the same loop and cannot
    interleave between the attribute reads — torn reads are not
    possible without an ``await`` boundary in this function body.
    """
    # ``getattr`` (rather than direct attribute access) so the function
    # tolerates partial mocks that omit ``_protocol`` entirely — the
    # acquire-path pre-ping calls this on every dequeued conn, including
    # ones built by tests that don't set up a protocol.
    protocol = getattr(conn, "_protocol", None)
    if protocol is None:
        return True
    # Poisoned-decoder short-circuit: a wire desync means the next
    # ROLLBACK round-trip would just be one more wasted RTT before
    # _read_response raises ProtocolError → invalidate. Drop the
    # connection now via the public is_wire_coherent accessor so the
    # pool does not chase a doomed reset. Wrap the access in the same
    # narrow exception filter as the transport-checks below to stay
    # safe against test mocks with a partial protocol shape.
    try:
        coherent = protocol.is_wire_coherent
    except (AttributeError, RuntimeError):
        coherent = True
    if isinstance(coherent, bool) and not coherent:
        return True
    writer = getattr(protocol, "_writer", None)
    reader = getattr(protocol, "_reader", None)
    transport = getattr(writer, "transport", None) if writer is not None else None
    # Narrow suppression to the specific categories a mock / partially
    # torn-down transport can legitimately raise. Wider `except Exception`
    # would mask programmer bugs (e.g. a misnamed attribute introduced by a
    # refactor). Mirrors the precedent set by ``_cleanup_loop_thread`` in
    # ``dqlitedbapi.connection``.
    try:
        closing = transport.is_closing() if transport is not None else False
    except (AttributeError, RuntimeError):
        closing = False
    try:
        eof = reader.at_eof() if reader is not None else False
    except (AttributeError, RuntimeError):
        eof = False
    return (isinstance(closing, bool) and closing) or (isinstance(eof, bool) and eof)


def _pool_unclosed_warning(
    closed_flag: list[bool],
    reserved_flag: list[bool],
    queue: "asyncio.Queue[DqliteConnection] | None" = None,
    creator_pid: int | None = None,
) -> None:
    """Emit a ``ResourceWarning`` when a ``ConnectionPool`` is GC'd
    without ``await close()``.

    Pool owns up to ``max_size`` live transports plus loop-bound
    asyncio primitives. A user that drops the pool reference
    without closing leaks the queued connections — this finalizer
    surfaces the leak as a driver-attributable diagnostic.

    Three-flag gate:

    - ``get_current_pid() != creator_pid``: forked child. The
      ``(closed_flag, reserved_flag, queue)`` snapshots belong to
      the parent process at fork-time; emitting a ResourceWarning
      here would falsely accuse the child of leaking what the
      parent owns. Mirrors the dbapi sibling's discipline at
      ``aio/connection.py::_async_unclosed_warning`` and the per-
      connection sibling at
      ``connection.py::_connection_unclosed_warning``. Without
      this gate, a forked worker that inherited a parent's open
      pool would emit one warning for the pool AND one for each
      queued ``DqliteConnection`` — an N+1 cascade of false
      positives. ``creator_pid=None`` keeps the helper callable
      from positional-only call sites that pre-date the pid guard
      (the helper is exported via ``__all__`` indirectly through
      the test suite).
    - ``closed_flag[0]`` flipped True by ``close()`` so orderly
      shutdown skips the warning.
    - ``reserved_flag[0]`` flipped True the first time the pool
      reserves a slot (either via ``initialize()`` warm-up OR
      ``acquire()``'s lazy-reservation arm). A pool that was
      constructed and dropped without ever acquiring has nothing
      to clean up — the warning would be a false positive.

    The ``queue`` parameter (when supplied) lets the warning report
    the queued-connection count so an operator who sees N+1 warnings
    (one from the pool, one per queued ``DqliteConnection``) can
    correlate them to a single root cause: the pool was dropped
    without ``await close()``. Per-conn warnings remain (every leaked
    resource gets its own warning per Python convention) but the
    pool's message now states the count up front. This is the
    minimum-change fix preferred by reviewer over a private-deque
    hop or a parallel-set bookkeeping rewrite.
    """
    if creator_pid is not None and get_current_pid() != creator_pid:
        # Forked child. Skip — the parent owns the lifecycle.
        return
    if closed_flag[0] or not reserved_flag[0]:
        return
    queued_count: int | None = None
    if queue is not None:
        with contextlib.suppress(Exception):
            queued_count = queue.qsize()
    if queued_count is not None and queued_count > 0:
        message = (
            "ConnectionPool was garbage-collected without await close(); "
            f"{queued_count} queued connection(s) will each emit their "
            "own ResourceWarning at GC for the same root cause. Call "
            "``await pool.close()`` explicitly to release them promptly."
        )
    else:
        message = (
            "ConnectionPool was garbage-collected without await close(); "
            "queued connections may have leaked their transports. Call "
            "``await pool.close()`` explicitly to release them promptly."
        )
    with contextlib.suppress(RuntimeError):
        warnings.warn(
            message,
            ResourceWarning,
            stacklevel=2,
        )


class ConnectionPool:
    """Connection pool with automatic leader detection.

    Thread safety: this class is NOT thread-safe. All operations must be
    performed within a single asyncio event loop. Do not share pool
    instances across OS threads or event loops. To submit work from other
    threads, use ``asyncio.run_coroutine_threadsafe()`` — the coroutines
    execute safely in the event loop thread. Free-threaded Python (no-GIL)
    is not supported.
    """

    def __init__(
        self,
        addresses: Sequence[str] | None = None,
        *,
        database: str = "default",
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 10.0,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        cluster: ClusterClient | None = None,
        node_store: NodeStore | None = None,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = 0.5,
        max_attempts: int | None = None,
        max_elapsed_seconds: float | None = None,
        dial_func: DialFunc | None = None,
    ) -> None:
        """Initialize connection pool.

        Args:
            addresses: List of node addresses. Ignored if ``cluster`` or
                ``node_store`` is provided; required otherwise.
            database: Database name
            min_size: Minimum connections to maintain
            max_size: Maximum connections allowed
            timeout: Per-RPC-phase timeout (forwarded to each pooled
                ``DqliteConnection``). Each phase of an operation
                (send, read, any continuation drain) gets the full
                budget independently, so a single call can take up to
                roughly N × ``timeout`` end-to-end. Wrap callers in
                ``asyncio.timeout(...)`` to enforce a wall-clock
                deadline. Acts as the default for ``dial_timeout``
                and ``attempt_timeout`` when those are ``None``.
            dial_timeout: Per-dial TCP-establish budget. Forwarded to
                the auto-built :class:`ClusterClient` and to every
                pooled :class:`DqliteConnection`. Defaults to
                ``timeout`` when ``None``. Mirrors go-dqlite's
                ``Config.DialTimeout``. Must be ``> 0`` when set.
            attempt_timeout: Per-attempt envelope (dial + handshake +
                first useful round-trip). Forwarded the same way.
                Defaults to ``timeout`` when ``None``. Mirrors
                go-dqlite's ``Config.AttemptTimeout``. Must be
                ``> 0`` when set.
            cluster: Externally-owned ClusterClient. Lets multiple pools
                share one ClusterClient (and thus its node store, leader
                cache, etc.) across databases or tenants.

                Ownership: the pool does NOT take ownership of this
                cluster. ``ClusterClient`` holds no long-lived resources
                (each probe opens a short-lived socket), so there is
                nothing to close on it. ``pool.close()`` drains pool-
                owned connections only.

                Sharing: one ClusterClient may back multiple pools
                concurrently. Direct use of the cluster (e.g.,
                ``await cluster.find_leader()``) from outside the pool is
                safe — each call opens a short-lived leader-query socket
                and does not contend with pool checkout.
            node_store: Externally-owned NodeStore used to build a new
                ClusterClient. Mutually exclusive with ``cluster``. Note
                that ``pool.close()`` does not close this auto-constructed
                cluster either — leader-query sockets it opens are
                short-lived and close on their own; the NodeStore itself
                is caller-owned.
            max_total_rows: Cumulative row cap across continuation
                frames for a single query. Forwarded to every
                :class:`DqliteConnection` the pool hands out, so every
                connection inherits the same governor. ``None`` disables
                the cap entirely (not recommended in production —
                bounds memory against slow-drip attacks).
            max_continuation_frames: Per-query continuation-frame cap.
                Complements ``max_total_rows``: a server
                sending 1-row-per-frame can inflict O(n) Python decode
                work where n is the row cap; the frame cap bounds that.
                Forwarded to every :class:`DqliteConnection`.
            trust_server_heartbeat: When True, the per-read deadline on
                every connection widens to the server-advertised
                heartbeat (up to a 300 s hard cap). Default False —
                operator-configured ``timeout`` is authoritative and
                the server cannot amplify it.
            close_timeout: Budget (seconds) for the transport-drain
                during ``close()``. After ``writer.close()`` the
                local side of the socket is gone; ``wait_closed`` is
                best-effort cleanup. The 0.5s default is sized for
                LAN; increase for WAN deployments where FIN/ACK
                round-trip is slower, or decrease to tighten
                SIGTERM-shutdown budgets. See
                ``DqliteConnection.__init__`` for full rationale.
            max_attempts: Maximum leader-discovery attempts per
                ``_create_connection`` (forwarded to
                ``ClusterClient.connect``). ``None`` (default) uses the
                cluster client's default of 3 — covers one leader
                change plus one transport hiccup. Increase only with
                eyes open: hiding genuine cluster instability behind
                a long retry loop just delays the diagnosis. Must be
                ``>= 1`` if not ``None``.
            max_elapsed_seconds: Total wall-clock cap on the
                per-connect retry loop (forwarded to
                ``ClusterClient.connect``). ``None`` (default) means
                only ``max_attempts`` governs termination. Set to a
                positive finite number for go-dqlite-style total-time
                bounding — composes with ``max_attempts``; whichever
                bound trips first ends the loop.
            dial_func: Optional caller-supplied dialer
                (:data:`dqliteclient.DialFunc`) replacing the default
                TCP path on every dial site. Forwarded to the
                auto-built :class:`ClusterClient` (and through to
                every pooled :class:`DqliteConnection`). Mutually
                exclusive with ``cluster=``: an externally-owned
                cluster already carries its own ``dial_func``, so
                supplying both raises ``ValueError`` to avoid silent
                divergence between the two paths. Mirrors go-dqlite's
                ``WithDialFunc``.
        """
        # Reject ``bool`` first: ``True == 1`` and ``False == 0``
        # would silently coerce to valid sizes and mask caller bugs
        # that accidentally pass a flag through. Mirrors the
        # int/bool reject discipline applied to ``encode_int64``,
        # ``Cursor.arraysize``, and the wire validators.
        if isinstance(min_size, bool) or not isinstance(min_size, int):
            raise TypeError(f"min_size must be int, got {type(min_size).__name__}")
        if isinstance(max_size, bool) or not isinstance(max_size, int):
            raise TypeError(f"max_size must be int, got {type(max_size).__name__}")
        if max_attempts is not None and (
            isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
        ):
            raise TypeError(f"max_attempts must be int or None, got {type(max_attempts).__name__}")
        if min_size < 0:
            raise ValueError(f"min_size must be non-negative, got {min_size}")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) must not exceed max_size ({max_size})")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1 if provided, got {max_attempts}")
        # ``max_elapsed_seconds`` validation parallels the
        # ``ClusterClient.connect`` validator: bool rejected, finite
        # positive only. Wording mirrors ``retry.py`` exactly so any
        # caller-side parsing of the message stays consistent across
        # layers.
        if max_elapsed_seconds is not None:
            if isinstance(max_elapsed_seconds, bool) or not isinstance(
                max_elapsed_seconds, (int, float)
            ):
                raise TypeError(
                    f"max_elapsed_seconds must be a number or None, "
                    f"got {type(max_elapsed_seconds).__name__}"
                )
            if not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0:
                raise ValueError(
                    f"max_elapsed_seconds must be a positive finite number, "
                    f"got {max_elapsed_seconds}"
                )
        validate_timeout(timeout)
        validate_timeout(
            close_timeout,
            name="close_timeout",
            min_value=0.01,
            min_value_rationale=CLOSE_TIMEOUT_FLOOR_RATIONALE,
        )
        if dial_timeout is not None:
            validate_timeout(dial_timeout, name="dial_timeout")
        if attempt_timeout is not None:
            validate_timeout(attempt_timeout, name="attempt_timeout")
        if cluster is not None and node_store is not None:
            raise ValueError("pass only one of cluster= or node_store=")
        if cluster is None and node_store is None and not addresses:
            raise ValueError("pass one of addresses, cluster, or node_store")
        # ``cluster=`` already carries its own ``dial_func``; supplying
        # both here would silently desync the pool's intended dialer
        # from the cluster's actual dialer (every pool dial goes
        # through ``self._cluster``). Reject explicitly.
        if cluster is not None and dial_func is not None:
            raise ValueError(
                "dial_func cannot be combined with cluster=; "
                "configure dial_func on the externally-owned ClusterClient"
            )

        self._addresses = addresses or []
        self._database = database
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._dial_timeout = dial_timeout
        self._attempt_timeout = attempt_timeout
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._trust_server_heartbeat = trust_server_heartbeat
        self._close_timeout = close_timeout
        self._max_attempts = max_attempts
        self._max_elapsed_seconds = max_elapsed_seconds

        if cluster is not None:
            # Externally-owned cluster — caller-supplied; pool does
            # not override its dial/attempt timeouts.
            self._cluster = cluster
        elif node_store is not None:
            self._cluster = ClusterClient(
                node_store,
                timeout=timeout,
                dial_timeout=dial_timeout,
                attempt_timeout=attempt_timeout,
                max_total_rows=max_total_rows,
                max_continuation_frames=max_continuation_frames,
                trust_server_heartbeat=trust_server_heartbeat,
                dial_func=dial_func,
            )
        else:
            self._cluster = ClusterClient.from_addresses(
                self._addresses,
                timeout=timeout,
                dial_timeout=dial_timeout,
                attempt_timeout=attempt_timeout,
                max_total_rows=max_total_rows,
                max_continuation_frames=max_continuation_frames,
                trust_server_heartbeat=trust_server_heartbeat,
                dial_func=dial_func,
            )
        self._pool: asyncio.Queue[DqliteConnection] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._closed_event: asyncio.Event | None = None
        self._close_done: asyncio.Event | None = None
        # Distinguish "first caller exited the drain phase" (signalled
        # via ``_close_done.set()`` in the close finally, so siblings
        # do not deadlock under cancel) from "drain ran to completion".
        # A second caller parking on ``_close_done.wait()`` and observing
        # the event set must NOT assume the queue was drained — the
        # first caller may have been interrupted mid-drain by an outer
        # cancel. Re-checking this flag after wait() returns lets the
        # second caller run a best-effort sweep so the documented
        # "queue drained" promise is upheld.
        self._drain_complete: bool = False
        self._initialized = False
        # Fork-after-init is unsupported: pooled connections hold
        # shared TCP sockets and asyncio primitives bound to the
        # parent's loop. Store the creator pid so cross-fork
        # ``acquire`` raises a clear ``InterfaceError`` instead of
        # silently corrupting the wire by interleaving writes.
        # Symmetric with ``__reduce__`` and the per-connection guard.
        self._creator_pid = os.getpid()
        # ResourceWarning finalizer mirrors the dbapi-layer pattern.
        # The flags are mutable cells so the finalizer reads the
        # latest values even after ``close()`` has run.
        # ``_reserved_flag`` flips True the first time a slot is
        # reserved (initialize() warm-up OR acquire()'s lazy-
        # reservation arm) — a never-acquired pool stays silent at
        # GC.
        self._closed_flag: list[bool] = [False]
        self._reserved_flag: list[bool] = [False]
        # Pass the queue through so the warning message reports the
        # queued-conn count. The queue stays alive until the
        # finalizer runs (it's stored in the finalize callback's
        # arg tuple), so qsize() is readable at warn time. The
        # queue does NOT hold a reference back to the pool — no
        # cycle is introduced.
        self._finalizer: weakref.finalize[Any, Any] | None = weakref.finalize(
            self,
            _pool_unclosed_warning,
            self._closed_flag,
            self._reserved_flag,
            self._pool,
            self._creator_pid,
        )

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has been called on this pool.

        Mirrors ``Connection.closed`` / ``Cursor.closed`` discipline
        used by every other connection-shaped class in the dbapi
        siblings (and by psycopg / asyncpg / aiosqlite). A
        long-running supervisor that owns the pool reference can use
        ``if not pool.closed: await pool.close()`` rather than
        reading the private ``_closed`` flag.

        Idempotent: ``close()`` on an already-closed pool is a silent
        no-op so reading this flag is the cheap predicate to skip the
        redundant call.
        """
        return self._closed

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        # Address list is part of the public configuration (no secrets
        # on dqlite's wire today); show the first few addresses with a
        # count hint so the repr stays short for large clusters.
        addrs = self._addresses[:3]
        suffix = f"+{len(self._addresses) - 3}" if len(self._addresses) > 3 else ""
        addrs_repr = f"{addrs!r}{suffix}"
        return (
            f"ConnectionPool(addresses={addrs_repr}, "
            f"size={self._size}, min_size={self._min_size}, "
            f"max_size={self._max_size}, {state})"
        )

    def __reduce__(self) -> "NoReturn":
        # ``asyncio.Queue`` and ``asyncio.Lock`` became pickleable in
        # Python 3.10+, so a naive ``pickle.dumps(pool)`` SILENTLY
        # produces a "live"-looking duplicate — detached from any
        # running loop, with fresh internal locks and queue. Any use
        # of the duplicate yields opaque corruption. Surface a clear
        # driver-level TypeError instead. Symmetric with the dbapi
        # Connection / Cursor guards.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — holds "
            f"loop-bound asyncio.Queue / asyncio.Lock / asyncio.Event "
            f"and live worker tasks; reconstruct from configuration "
            f"in the target process instead."
        )

    async def initialize(self) -> None:
        """Initialize the pool with minimum connections.

        Idempotent: concurrent callers share the same initialization —
        only one performs the TCP work, the others await its result.

        Partial-failure behavior: ``asyncio.gather`` with the
        default ``return_exceptions=False`` cancels sibling tasks on
        first failure but does NOT close connections that already
        succeeded — they leak as orphaned transports. Use
        ``return_exceptions=True`` so every task resolves, then close
        survivors explicitly before re-raising the first failure.

        Warm-up shape: this issues ``min_size`` connection handshakes
        IN PARALLEL via ``asyncio.gather``. Every initial connect
        targets the current leader (after cluster discovery), so the
        leader-side handshake acceptance is serialised and a large
        ``min_size`` does NOT linearly speed up startup. Diverges from
        Go-dqlite's pool which warms lazily; the parallel-warm shape
        is a deliberate trade-off: predictable cold-start memory at
        the cost of leader-side serialisation. Keep ``min_size`` low
        (single digits) unless steady-state concurrency demands warm
        connections at engine startup.
        """
        if _conn_mod._current_pid != self._creator_pid:
            raise InterfaceError(
                f"Pool used after fork; reconstruct from configuration "
                f"in the target process. (created in pid {self._creator_pid}, "
                f"current pid {_conn_mod.get_current_pid()})"
            )
        # Hold the lock across the gather so a second concurrent
        # initialize() call observes _initialized=True after the first
        # completes and returns without re-creating.
        async with self._lock:
            if self._initialized:
                return
            if self._min_size > 0:
                logger.debug("pool.initialize: requesting %d connections", self._min_size)
                self._size += self._min_size
                if self._size > 0:
                    self._reserved_flag[0] = True
                # Count reservations that still need to be released. Each
                # successful put into the pool queue "commits" one slot
                # (the connection stays and must remain counted in
                # _size), so unqueued shrinks per iteration. On any
                # abort the finally below releases exactly ``unqueued``
                # slots — the ones that never made it to the queue —
                # and closes the unqueued survivors.
                unqueued = self._min_size
                unqueued_survivors: list[DqliteConnection] = []
                # Build child tasks explicitly (not via the ``gather`` *
                # expression form) so a CancelledError raised out of
                # ``gather`` itself — outer ``asyncio.timeout`` / a
                # TaskGroup sibling failure / ``task.cancel()`` —
                # leaves the finally with a handle on every child's
                # ``_result``. The expression form discards the
                # already-completed children's results into the
                # gather's local frame; once gather raises, those
                # results are unreachable and the live conns are
                # orphaned (transports leaked until GC).
                #
                # This is the symmetric resource-discipline fix to
                # the prior ``_size`` accounting fix that moved the
                # bookkeeping into the finally; the connection-close
                # sweep now does the same by walking the explicit
                # task list.
                # Build ``create_tasks`` INSIDE the try frame so a
                # BaseException landing mid-construction (synthetic
                # KeyboardInterrupt, outer cancel in the bytecode
                # window before the ``try:``) keeps every
                # already-created task tracked: the ``finally:`` walks
                # the partial list via the ``gather_returned`` flag
                # below, cancels survivors, and closes their
                # connections. Pre-fix the comprehension ran outside
                # the try frame; a BaseException there orphaned the
                # live tasks (loop-bound primitives and transports
                # leaked until GC). Mirrors the cluster-side hardening
                # at ``_find_leader_impl`` and the pool-acquire
                # discipline applied earlier on the acquire path that
                # established the project pattern of building the task
                # set inside the try frame whose finally cancels and
                # gathers them.
                create_tasks: list[asyncio.Task[DqliteConnection]] = []
                # Track whether ``gather`` returned normally; if it
                # raised CancelledError, the post-gather assignment
                # to ``unqueued_survivors`` (and the put-loop's pop)
                # never ran, so the finally must walk the explicit
                # task list to recover completed children's
                # results. When gather DID return normally, the
                # existing pop-based discipline already tracked the
                # un-queued tail precisely; walking tasks again
                # would re-add already-queued conns and double-close.
                gather_returned = False
                try:
                    for _ in range(self._min_size):
                        create_tasks.append(asyncio.create_task(self._create_connection()))
                    # Create min_size connections concurrently so startup
                    # latency doesn't scale with min_size × per-connect RTT.
                    results = await asyncio.gather(
                        *create_tasks,
                        return_exceptions=True,
                    )
                    gather_returned = True
                    successes: list[DqliteConnection] = []
                    failures: list[BaseException] = []
                    for r in results:
                        if isinstance(r, BaseException):
                            failures.append(r)
                        else:
                            successes.append(r)
                    if failures:
                        # Close the connections that did succeed — they
                        # are unowned now that initialize is aborting.
                        # ``%r`` on a server-supplied exception would
                        # echo ``OperationalError.message`` / wrapped
                        # peer text into the log without sanitisation.
                        # Route ``str(exc)`` (not ``repr(exc)``) through
                        # ``sanitize_for_log`` so the wire-layer
                        # single-stage substitution (``?`` for control /
                        # bidi / invisible, ``\n`` / ``\t`` literal-
                        # escape for LF / Tab) lands on the operator-
                        # facing text directly — ``repr()`` would have
                        # double-encoded the same characters through
                        # Python's escape sequences, producing harder-
                        # to-read log lines than the sibling
                        # ``sanitize_for_log(str(...))`` sites use.
                        # Prepend ``type(exc).__name__`` so the class-
                        # context cue that ``repr()`` provided survives
                        # the switch.
                        _first_failure = failures[0]
                        logger.debug(
                            "pool.initialize: aborting after %d/%d creates succeeded; "
                            "closing %d survivors (first failure: %s: %s)",
                            len(successes),
                            self._min_size,
                            len(successes),
                            type(_first_failure).__name__,
                            sanitize_for_log(str(_first_failure)),
                        )
                        # Log every failure individually so operators
                        # see the full picture before unpacking the
                        # ExceptionGroup — partial cluster outages
                        # frequently produce different root causes per
                        # node (timeout / refused / no-leader) and the
                        # single-exception surface hid the majority
                        # signal.
                        for i, exc in enumerate(failures):
                            logger.warning(
                                "pool.initialize: create_connection %d/%d failed: %s: %s",
                                i + 1,
                                len(failures),
                                type(exc).__name__,
                                sanitize_for_log(str(exc)),
                            )
                        for conn in successes:
                            try:
                                await conn.close()
                            except _POOL_CLEANUP_EXCEPTIONS:
                                logger.debug(
                                    "pool.initialize: partial-cleanup close error",
                                    exc_info=True,
                                )
                        # Preserve the single-failure narrow type so
                        # callers doing ``except DqliteConnectionError``
                        # continue to match. For multiple distinct
                        # failures, raise an ExceptionGroup so every
                        # cause is accessible via structured handling
                        # (``except*``); the finally releases the
                        # reservation before the raise propagates.
                        if len(failures) == 1:
                            raise failures[0]
                        # Bounded aggregate: at most
                        # ``_MAX_AGGREGATE_CHILDREN`` children so the
                        # exception graph stays picklable for cross-
                        # process error capture (Celery, multiprocessing).
                        from dqliteclient.cluster import _bounded_group

                        raise _bounded_group(
                            f"pool.initialize: {len(failures)} of {self._min_size} connects failed",
                            failures,
                        )
                    # Track which successes are still unqueued so a
                    # cancellation mid put-loop can close them precisely
                    # rather than leaking their transports.
                    unqueued_survivors = list(successes)
                    for conn in successes:
                        # Re-check _closed every iteration: close() does
                        # not acquire _lock, so a concurrent close()
                        # landing while gather was suspended would
                        # otherwise let us commit connections into a
                        # pool whose _closed flag is True. In that case
                        # acquire() refuses them (they are invisible to
                        # the user) and the transports leak. Route the
                        # tail through the existing unqueued_survivors
                        # cleanup path instead.
                        if self._closed:
                            break
                        await self._pool.put(conn)
                        # put() succeeded: the slot is now committed —
                        # the connection stays in _size accounting and
                        # belongs to the queue, not to this function.
                        unqueued -= 1
                        unqueued_survivors.pop(0)
                    logger.debug("pool.initialize: %d connections ready", self._min_size)
                finally:
                    # Any exit path with uncommitted reservations —
                    # failed gather, raise from _pool.put, outer
                    # CancelledError mid put-loop — must return the
                    # unqueued slots to _size so a subsequent
                    # initialize()/acquire() is not blocked against
                    # a stale counter climbing toward _max_size.
                    # Route through the same under-flow-guarded helper
                    # as the per-conn ``_release_reservation`` path so
                    # a future double-decrement at this site lands the
                    # canonical ERROR log instead of silently producing
                    # a negative ``_size``. The lock is already held
                    # by this method's outer ``async with self._lock``.
                    if unqueued > 0 and self._release_reservations_locked(unqueued):
                        self._signal_state_change()
                    # Close any connection that made it past the gather
                    # but never into the queue. Under a clean success
                    # this list is empty; on partial put-loop cancel it
                    # holds the unqueued tail.
                    #
                    # Cancel-during-gather recovery: ``gather``
                    # propagates CancelledError when its caller is
                    # cancelled, but children that already completed
                    # have their results sitting in their tasks'
                    # ``_result`` slot. The post-gather assignment to
                    # ``unqueued_survivors`` did not run, so this
                    # loop iterates an empty list. Walk
                    # ``create_tasks`` to recover those results.
                    # Already-failed / not-yet-done / cancelled tasks
                    # are skipped — only successful results need
                    # closing here. Only walks the task list if
                    # ``gather`` raised before returning; once the
                    # post-gather code ran, the pop-based
                    # discipline already tracked the un-queued tail
                    # precisely (walking again would double-close
                    # conns already in the queue).
                    if not gather_returned:
                        # Cancel any task that's still pending (e.g.
                        # BaseException landed mid-task-creation before
                        # gather ran, or gather itself was cancelled
                        # mid-flight) and await them so their slots
                        # don't surface as orphan-task warnings at GC.
                        # Without this pass, a pending task created
                        # inside this frame outlives the function
                        # frame with no observer.
                        cancelled_pending: list[asyncio.Task[DqliteConnection]] = []
                        for t in create_tasks:
                            if not t.done():
                                t.cancel()
                                cancelled_pending.append(t)
                        # Pass-1 gather can ITSELF be cancelled by an
                        # outer ``BaseException`` (e.g. an
                        # ``asyncio.timeout`` wrapping initialize()
                        # firing inside this gather). Without the
                        # try/except, the gather re-raises and pass-2
                        # is skipped — connections produced by tasks
                        # that completed JUST BEFORE being cancelled
                        # are orphaned (transports leaked → GC
                        # ResourceWarning). Wrap the gather and route
                        # to the same close-completed helper from
                        # both arms so the recovery walk runs even
                        # under outer cancel; re-raise after closing
                        # so structured concurrency still propagates.
                        outer_cancel: BaseException | None = None
                        if cancelled_pending:
                            try:
                                await asyncio.gather(*cancelled_pending, return_exceptions=True)
                            except BaseException as exc:
                                outer_cancel = exc
                        self._initialize_collect_completed_conns(create_tasks, unqueued_survivors)
                        if outer_cancel is not None:
                            # Close before re-raising so the caller's
                            # outer cancel doesn't orphan the survivors.
                            await self._initialize_close_unqueued(unqueued_survivors)
                            raise outer_cancel
                    await self._initialize_close_unqueued(unqueued_survivors)
            # Do not mark initialized if close() landed during the
            # put-loop and we broke out early — otherwise a subsequent
            # initialize() call on a (re-opened) pool short-circuits.
            if not self._closed:
                self._initialized = True

    @staticmethod
    def _initialize_collect_completed_conns(
        create_tasks: list[asyncio.Task[DqliteConnection]],
        unqueued_survivors: list[DqliteConnection],
    ) -> None:
        """Append connections from completed-and-not-cancelled tasks
        in ``create_tasks`` to ``unqueued_survivors``.

        Factored out of ``initialize``'s finally so the
        cancel-during-pass-1 recovery arm and the normal post-gather
        recovery arm walk the same code — both paths must stay in
        lockstep so a future divergence in the close-discipline
        cannot orphan completed-task connections.
        """
        for t in create_tasks:
            if not t.done() or t.cancelled():
                continue
            try:
                r = t.result()
            except BaseException:
                continue
            unqueued_survivors.append(r)

    async def _initialize_close_unqueued(
        self,
        unqueued_survivors: list[DqliteConnection],
    ) -> None:
        """Close every connection in ``unqueued_survivors``.

        Each close is shielded individually so an outer cancel during
        the walk doesn't orphan the remaining conns. ``conn.close``
        legitimately raises any exception in
        ``_POOL_CLEANUP_EXCEPTIONS`` on a partially-torn-down
        transport — those land in DEBUG; anything else propagates.
        """
        for conn in unqueued_survivors:
            try:
                await asyncio.shield(conn.close())
            except _POOL_CLEANUP_EXCEPTIONS:
                logger.debug(
                    "pool.initialize: unqueued-survivor close error",
                    exc_info=True,
                )
            except asyncio.CancelledError:
                # Outer cancel propagated through the shielded close
                # boundary (the close still completes via the shield).
                # Continue closing the remaining survivors before
                # letting the cancel resume.
                logger.debug(
                    "pool.initialize: cancel during unqueued-survivor close",
                    exc_info=True,
                )

    async def _create_connection(self) -> DqliteConnection:
        """Create a new connection to the leader.

        Does NOT mutate ``self._size`` — callers must have incremented
        ``_size`` under ``_lock`` as a reservation before calling this
        method (see ``acquire`` / ``initialize``) so that concurrent
        callers cannot collectively exceed ``_max_size``. On failure,
        the caller is responsible for decrementing to release the
        reservation.
        """
        return await self._cluster.connect(
            database=self._database,
            max_total_rows=self._max_total_rows,
            max_continuation_frames=self._max_continuation_frames,
            trust_server_heartbeat=self._trust_server_heartbeat,
            close_timeout=self._close_timeout,
            max_attempts=self._max_attempts,
            max_elapsed_seconds=self._max_elapsed_seconds,
        )

    async def _put_back_or_release_late_winner(self, conn: DqliteConnection) -> None:
        """Put a connection back on the queue, or close + release the
        reservation if the queue is full.

        Used in two places in ``acquire()``:

        1. The ``except BaseException`` arm — outer cancel raced with
           a successful ``get_task`` (a sibling ``_release``
           ``put_nowait`` ran between our snapshot and the cancel).
        2. The post-wait demux's else-arm — timeout snapshot raced a
           winning ``get_task`` during the post-wait
           ``await closed_task``.

        Without this routing, the conn is referenced only by the
        soon-to-be-GC'd ``get_task`` and silently disappears. Its
        reservation slot is never released because ``_release`` only
        fires for connections that flow back through the user's
        context manager — so the pool permanently loses one slot of
        capacity per occurrence.
        """
        # If pool.close() ran between the snapshot and this re-route,
        # putting the conn back into the queue would leak it: the
        # drain runs once per close() (gated by _drain_complete), and
        # a put_nowait after the drain finishes parks the conn in a
        # queue no one will revisit. Close the conn directly and
        # release the reservation so _size stays consistent. Mirrors
        # the sibling closed-pool short-circuit in ``_release`` —
        # ``done/client-pool-close-no-recheck-before-put-nowait-leaks-conn.md``
        # established the same discipline for the _release path.
        if self._closed:
            # Clear ``_pool_released`` BEFORE close so the close path
            # actually runs. ``DqliteConnection.close()`` early-returns
            # at the ``if self._pool_released: return`` guard when the
            # flag is True — set by the prior ``_release`` write
            # before the put_nowait that this late-winner conn came
            # out of. Without the flip the close becomes a no-op and
            # the transport / reader task leak while
            # ``_release_reservation`` decrements as though the close
            # had happened. Mirrors the ``_drain_idle`` discipline.
            conn._pool_released = False
            try:
                with contextlib.suppress(OSError):
                    await asyncio.shield(conn.close())
            finally:
                # Restore the flag for contract symmetry with
                # ``_drain_idle``: any stale-reference second close()
                # falls through the documented early-return rather
                # than re-running close on a dead conn.
                conn._pool_released = True
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._release_reservation())
            return
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Invariant violation: reservations should track queue
            # capacity exactly, so a full queue on return is
            # "impossible." If it happens anyway, silently dropping
            # the reference would leak a live reader task and a
            # socket. Close explicitly and adjust the reservation
            # count so the pool shrinks cleanly instead of leaking.
            # Suppression of close's own errors is narrow — OSError on
            # an already-dead writer is expected; anything else
            # propagates. Flip ``_pool_released`` to ``False`` first
            # so the close actually runs (see the closed-pool arm
            # above for the rationale).
            conn._pool_released = False
            try:
                with contextlib.suppress(OSError):
                    await asyncio.shield(conn.close())
            finally:
                conn._pool_released = True
            # Route through the helper so the counter stays
            # lock-protected and sibling acquirers parked on
            # ``closed_event.wait()`` get woken via
            # ``_signal_state_change``. Shield so a nested cancel
            # cannot leave ``_size`` inconsistent.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._release_reservation())

    def _release_reservations_locked(self, n: int) -> bool:
        """Helper: decrement ``_size`` by ``n`` (>= 1) with the
        under-flow guard, assuming the caller already holds
        ``self._lock``.

        Returns ``True`` if the decrement actually happened (caller
        should signal state-change), ``False`` if the under-flow
        guard refused. Centralises the underflow-guard log so every
        ``_size -= n`` site goes through the same diagnostic path —
        callers don't need to re-implement the guard, and a future
        double-release at any site lands the canonical ERROR log.

        Logs at ERROR if ``self._size < n`` (would underflow); refuses
        the decrement to keep accounting non-negative. Skipping the
        state-change signal on the refusal path is intentional — the
        refusal isn't a transition waiters need to react to.

        Validation:

        * ``n`` must be a positive ``int`` (``bool`` rejected
          explicitly: ``isinstance(True, int)`` is True). ``n <= 0``
          is rejected because ``n=0`` is a no-op decrement that the
          caller's ``_signal_state_change()`` would broadcast
          spuriously, and ``n<0`` would silently INCREMENT ``_size``
          (the underflow guard ``_size < n`` fails open against a
          negative ``n``) — exactly the symmetric corruption the
          guard was meant to prevent.
        * The lock-held precondition is checked at runtime (NOT via
          ``assert`` — bare ``assert`` is stripped under
          ``python -O``, and this is a real precondition contract on
          shared-state mutation, not a mypy narrow). A future caller
          forgetting the lock raises ``AssertionError`` immediately
          rather than corrupting ``_size`` under contention.
        """
        if not self._lock.locked():
            raise AssertionError("_release_reservations_locked called without _lock held")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"_release_reservations_locked requires n >= 1 (int), got {n!r}")
        if self._size < n:
            logger.error(
                "pool: _release_reservations_locked called with _size=%d, n=%d; "
                "ignoring to keep accounting non-negative. This indicates a "
                "double-release bug — check recent changes to the cancel/cleanup paths.",
                self._size,
                n,
            )
            return False
        self._size -= n
        return True

    async def _release_reservation(self) -> None:
        """Decrement ``_size`` by 1 under the lock, waking waiters.

        Every per-conn ``_size -= 1`` call in the pool must go
        through this helper so the counter stays consistent against
        concurrent capacity checks in ``acquire``. Bulk decrements
        (currently only in ``initialize`` recovery) call
        ``_release_reservations_locked(n)`` while holding the lock
        themselves — both paths share the same under-flow guard.
        """
        async with self._lock:
            decremented = self._release_reservations_locked(1)
        if decremented:
            self._signal_state_change()

    def _get_closed_event(self) -> asyncio.Event:
        """Lazily create the closed Event bound to the running loop.

        Every caller (``acquire`` at line ~434, ``_signal_state_change``
        for the non-None-event path) is gated by an ``if self._closed:
        raise`` check above, so entering this method with
        ``self._closed == True`` is not a reachable production state.
        ``close()`` sets the event directly when it exists rather than
        going through this path.
        """
        if self._closed_event is None:
            self._closed_event = asyncio.Event()
        return self._closed_event

    def _signal_state_change(self) -> None:
        """Wake any acquire() waiters to re-check pool state.

        Reuses the closed-event path so we don't add a second primitive.
        acquire() clears the event right before it parks each iteration,
        so set()s that come through this helper reliably wake the current
        wait; waiters always re-check _closed at the loop top.
        """
        if self._closed_event is not None:
            self._closed_event.set()

    async def _drain_idle(self) -> None:
        """Close all idle connections in the pool.

        Called when a connection is found to be broken (e.g., after a
        leader change or server restart), since other idle connections
        are likely stale too.
        """
        try:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover
                    # Defensive: the ``empty()`` check is racy with a
                    # concurrent ``get_nowait`` from another task. The
                    # window between check and get is too tight to drive
                    # without monkey-patching ``asyncio.Queue`` itself.
                    # Verified by code review, not coverage.
                    break
                try:
                    # Clear ``_pool_released`` BEFORE close so the
                    # close path actually runs. The release flag is
                    # set when the conn was put back into the queue
                    # in ``_release``; ``DqliteConnection.close()``
                    # early-returns on True (user-side close on a
                    # checked-in conn must be a no-op). Without this
                    # clear, the pool's drain-side close would be
                    # silently absorbed and the writer would leak.
                    conn._pool_released = False
                    # Shield each per-connection close against an outer
                    # ``asyncio.timeout(pool.close())``. Without the shield,
                    # a cancel that lands mid-``wait_closed`` propagates out
                    # of the drain and orphans every subsequent queued
                    # connection (its reader task + transport leak until GC
                    # prints ``"Task was destroyed but it is pending"`` at
                    # interpreter exit). The outer cancel still aborts the
                    # drain loop — but every connection that was STARTED on
                    # the close path finishes cleanly.
                    await asyncio.shield(conn.close())
                except Exception:
                    # Transport-level failures (BrokenPipeError, OSError, our
                    # own DqliteConnectionError) are absorbed so drain can
                    # finish the remaining connections. CancelledError /
                    # KeyboardInterrupt / SystemExit propagate — swallowing
                    # them used to break structured concurrency (``asyncio.
                    # timeout`` around ``pool.close()`` would silently hang).
                    logger.debug(
                        "pool: close() on idle connection %s failed",
                        sanitize_for_log(str(getattr(conn, "_address", "?"))),
                        exc_info=True,
                    )
                finally:
                    # Restore ``_pool_released`` so a stale-reference
                    # caller (e.g. a weakref proxy held from before
                    # the drain) attempting close() again hits the
                    # documented "user-side close on a checked-in
                    # conn is a no-op" early-return. Functionally
                    # safe today (``_protocol`` is None after close,
                    # so a second close falls through harmlessly),
                    # but the contract that "True ⇔ pool owns the
                    # close path" is restored — brittle to refactor
                    # otherwise.
                    conn._pool_released = True
                    # Shield the reservation decrement against outer cancel:
                    # an ``asyncio.timeout`` around ``pool.close()`` can fire
                    # during ``_release_reservation``'s lock acquire; without
                    # the shield, _size drifts above actual capacity. Sibling
                    # to the exception-path shield at ``acquire()``.
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
        finally:
            # Multi-connection-leak guard: the per-iteration shield
            # protects an in-flight close, but a cancel propagating out
            # of one iteration aborts the loop and leaves the remaining
            # queued connections un-closed. Drain whatever is still in
            # the queue under shield, best-effort, so no FakeConn /
            # DqliteConnection sits orphaned in the pool's queue after
            # ``close()`` returns. The original cancel still propagates
            # via the surrounding ``finally``.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._drain_remaining_after_cancel())

    async def _drain_remaining_after_cancel(self) -> None:
        """Best-effort sweep for connections still in the queue after
        the main drain loop exited (typically via outer cancel).

        Each remaining connection has its ``close()`` awaited and a
        reservation slot released, mirroring the main loop's
        bookkeeping. Failures here are absorbed so a single bad close
        does not abort the cleanup of the rest. Re-entry from a second
        ``close()`` caller is safe — the queue empties on the first
        sweep and subsequent calls find ``self._pool.empty() is True``.
        """
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover
                break
            try:
                # Shield the close so a KeyboardInterrupt / SystemExit
                # (or a fresh cancel delivered through the lock acquire
                # in close()) does not abort the drain mid-iteration
                # and leak the rest of the queue's transports +
                # reservations. The per-iteration release below uses
                # the same shield discipline.
                await asyncio.shield(conn.close())
            except Exception:
                logger.debug(
                    "pool: cleanup-after-cancel close() on %s failed",
                    sanitize_for_log(str(getattr(conn, "_address", "?"))),
                    exc_info=True,
                )
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Acquire a connection from the pool."""
        if _conn_mod._current_pid != self._creator_pid:
            raise InterfaceError(
                f"Pool used after fork; reconstruct from configuration "
                f"in the target process. (created in pid {self._creator_pid}, "
                f"current pid {_conn_mod.get_current_pid()})"
            )
        if self._closed:
            raise DqliteConnectionError(f"Pool is closed (id={id(self)})")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        conn: DqliteConnection | None = None

        while conn is None:
            if self._closed:
                raise DqliteConnectionError(f"Pool is closed (id={id(self)})")

            # Try to get an idle connection from the queue
            try:
                conn = self._pool.get_nowait()
                break
            except asyncio.QueueEmpty:
                pass

            # Try to reserve a new-connection slot under the lock, then
            # drop the lock before the TCP handshake so concurrent
            # pool users aren't serialized on network latency.
            #
            # ``try:`` opens BEFORE ``async with self._lock:`` so the
            # ``self._size += 1`` increment lives inside the same
            # try-frame whose except arm releases the reservation.
            # Without that envelope, a ``BaseException`` (KeyboardInterrupt,
            # synthetic post-lock-exit failure) landing between the
            # ``async with self._lock:`` exit and the ``try:`` start
            # would escape with the slot reserved but no compensating
            # decrement. Mirrors the ``DqliteConnection._run_protocol``
            # discipline (set-flag-INSIDE-try, not after).
            reserved = False
            try:
                async with self._lock:
                    if self._closed:
                        raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
                    if self._size < self._max_size:
                        self._size += 1
                        self._reserved_flag[0] = True
                        reserved = True
                if reserved:
                    # Clamp the create-connection await to the
                    # remaining acquire deadline. ``_create_connection``
                    # delegates to ``ClusterClient.connect`` whose
                    # internal retry budget can run for tens of
                    # seconds — far beyond the user-supplied
                    # ``pool.timeout``. Without this clamp,
                    # ``acquire(timeout=0.1)`` could block for the
                    # cluster.connect retry budget and only the
                    # message-rendered timeout in the queue-wait phase
                    # honoured the user contract. The dead-conn
                    # replacement arm below applies the same clamp.
                    create_remaining = deadline - loop.time()
                    if create_remaining <= 0:
                        raise TimeoutError
                    async with asyncio.timeout(create_remaining):
                        conn = await self._create_connection()
            except BaseException as exc:
                # Shield the release so an outer cancel re-arming
                # on the await checkpoint does not bypass the
                # decrement; without the shield, each
                # cancel-mid-create leaks one ``_size`` slot and
                # the pool eventually wedges at max_size with no
                # checked-out connections. ``if reserved:`` guards
                # against double-release on the pre-grant
                # ``_closed`` raise (which never reaches the
                # increment).
                if reserved:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
                if isinstance(exc, TimeoutError):
                    # The clamp fired before ``_create_connection``
                    # produced a usable conn (``TimeoutError`` is only
                    # raised inside the ``async with asyncio.timeout(
                    # create_remaining):`` scope, which is itself gated
                    # by ``if reserved:`` above; the pre-grant
                    # ``_closed`` raise is ``DqliteConnectionError``).
                    # Surface as the project's user-facing class with an
                    # actionable message naming the fresh-dial phase,
                    # mirroring the queue-wait timeout shape below so
                    # operators see a consistent diagnostic regardless
                    # of which phase exhausted the budget.
                    idle = self._pool.qsize()
                    checked_out = self._size - idle
                    raise DqliteConnectionError(
                        f"Timed out creating a fresh connection from the pool "
                        f"(pool_id={id(self)}, max_size={self._max_size}, "
                        f"checked_out={checked_out}, idle={idle}, "
                        f"timeout={self._timeout}s)."
                    ) from exc
                raise
            if reserved:
                # mypy: ``conn`` is assigned inside the ``if reserved:``
                # arm of the try-block above on the success path; the
                # except arm re-raises so any path arriving here with
                # ``reserved=True`` has a non-None ``conn``. The
                # explicit narrowing keeps the strict-mode union-attr
                # checks happy below without runtime cost on the happy
                # path.
                assert conn is not None
                # close() may have run while ``_create_connection``
                # was suspended on leader discovery / TCP handshake.
                # Without this re-check, the fresh connection would
                # be yielded on a pool whose ``_closed`` flag is True
                # — contract violation and a sneaky leak (user runs
                # real queries against an invisibly-closed pool until
                # they exit the ``async with`` block). The dead-conn-
                # replacement arm at lines 1196-1210 has the
                # symmetric discipline; this restores parity. Shield
                # both the close and the reservation release so an
                # outer cancel cannot leak the freshly-built
                # connection's transport or the reservation slot.
                if self._closed:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(conn.close())
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
                    raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
                break

            # At capacity — wait briefly on the queue, then loop back to
            # re-check capacity (another coroutine may have freed a slot)
            remaining = deadline - loop.time()
            if remaining <= 0:
                # Compute saturation snapshot once for the message —
                # ``_size`` is checked-out-plus-reserved; ``qsize`` is
                # idle. A pool at ``checked_out == max_size`` with
                # ``idle == 0`` is leaking; otherwise the cluster is
                # slow. The ``pool_id`` lets operators correlate the
                # failed acquire with the warm-up / drain log lines
                # that already include ``id(self)``.
                idle = self._pool.qsize()
                checked_out = self._size - idle
                raise DqliteConnectionError(
                    f"Timed out waiting for a connection from the pool "
                    f"(pool_id={id(self)}, max_size={self._max_size}, "
                    f"checked_out={checked_out}, idle={idle}, "
                    f"timeout={self._timeout}s). "
                    f"If checked_out is at max_size, the application is "
                    f"leaking connections; otherwise the cluster is slow."
                )
            # Race the queue against the state-change event so any pool
            # state change (close, size decrement, drain) wakes waiters
            # promptly. The check-_closed-then-clear pair runs under
            # the lock so a concurrent close() can't set() the event
            # between our read and our clear.
            async with self._lock:
                closed_event = self._get_closed_event()
                if self._closed:
                    raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
                closed_event.clear()
                logger.debug(
                    "pool.acquire: at capacity size=%d max=%d, waiting",
                    self._size,
                    self._max_size,
                )
            get_task: asyncio.Task[DqliteConnection] | None = None
            closed_task: asyncio.Task[bool] | None = None
            try:
                # Create both tasks inside the try so an outer
                # ``CancelledError`` between the two ``create_task`` calls
                # cannot orphan one of them — the finally below cancels
                # whichever task(s) exist.
                get_task = asyncio.create_task(self._pool.get())
                closed_task = asyncio.create_task(closed_event.wait())
                done, _pending = await asyncio.wait(
                    {get_task, closed_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                # Outer cancellation of this coroutine while suspended in
                # ``asyncio.wait``. ``asyncio.wait`` does not cancel its
                # argument tasks, so both children are still alive — we
                # MUST stop them before propagating, otherwise the
                # abandoned get_task can win a later queue.put() and
                # orphan a connection (silently shrinking pool capacity).
                if closed_task is not None and not closed_task.done():
                    closed_task.cancel()
                    # Symmetric with the happy-path cleanup at line ~554:
                    # await the cancelled task so the CancelledError
                    # doesn't sit on the task object until GC, which
                    # would emit "Task exception was never retrieved"
                    # under rapid cancel churn and keep a reference to
                    # ``closed_event`` alive via the task. Narrow to
                    # ``CancelledError`` so a ``KeyboardInterrupt`` /
                    # ``SystemExit`` raised by the cancelled task's
                    # underlying coroutine still propagates, per
                    # Python signal-propagation contract.
                    with contextlib.suppress(asyncio.CancelledError):
                        await closed_task
                if (
                    get_task is not None
                    and get_task.done()
                    and not get_task.cancelled()
                    and get_task.exception() is None
                ):
                    # Outer cancel raced with a successful get. The
                    # reservation that backed this connection is still
                    # valid; return it to the queue so the next
                    # acquirer can use it instead of closing and
                    # releasing (which would shrink _size).
                    await self._put_back_or_release_late_winner(get_task.result())
                elif get_task is not None and not get_task.done():
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                elif get_task is not None and get_task.done():
                    # Queue.get() completed but with cancellation or an
                    # exception. Consume the result so the task does not
                    # log "Task exception was never retrieved" at GC and
                    # so the task object can be released cleanly. Suppress
                    # BaseException because the consumed exception may
                    # itself be a CancelledError, and the outer cancel
                    # below takes priority.
                    with contextlib.suppress(BaseException):
                        await get_task
                raise
            # mypy-narrowing assert; no runtime cost on the happy path
            # (both tasks are constructed in the try-block above before
            # the BaseException branch raises). Stripped under
            # ``python -O`` but the invariant is structurally guaranteed.
            assert get_task is not None and closed_task is not None
            if not closed_task.done():
                closed_task.cancel()
                # Await the cancelled task so the CancelledError doesn't
                # sit on the task object until GC. asyncio.Event.wait()
                # exits cleanly on cancel so this returns promptly.
                # Symmetric with the get_task-cancel branch below.
                # Narrow to ``CancelledError``: a ``KeyboardInterrupt``
                # / ``SystemExit`` raised by the underlying coroutine
                # must still propagate.
                with contextlib.suppress(asyncio.CancelledError):
                    await closed_task
            if get_task.done() and not get_task.cancelled() and get_task.exception() is None:
                # Live-state check: ``done`` is the snapshot taken
                # before the post-wait ``await closed_task`` yield
                # above, during which a sibling ``_release`` can
                # ``put_nowait`` and resolve ``get_task``. Trusting
                # ``get_task in done`` would silently route the
                # winning conn into the cancel-and-discard arm,
                # leaking one slot of capacity per occurrence.
                conn = get_task.result()
            else:
                # Either close fired or the poll timer fired; either way,
                # cancel the queue wait cleanly and let the loop re-check.
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                if get_task.done() and not get_task.cancelled() and get_task.exception() is None:
                    # Cancel raced a successful get during the await
                    # above (a sibling ``_release`` put_nowait between
                    # our cancel call and the cancel actually
                    # delivering). Route the conn back via the same
                    # put-back-or-release path the outer-cancel arm
                    # uses, so the slot is not leaked.
                    await self._put_back_or_release_late_winner(get_task.result())
                continue

        # If connection is dead, discard and create a fresh one with leader discovery.
        # Also drain other idle connections — they likely point to the same dead server.
        #
        # ``is_connected`` is the protocol-level handshake-complete flag;
        # ``_socket_looks_dead`` is a non-blocking transport-level peek
        # (transport.is_closing() / reader.at_eof() / poisoned-decoder
        # short-circuit). An idle connection that has seen a clean peer
        # FIN (leader flip with graceful close) passes ``is_connected``
        # but trips ``_socket_looks_dead`` — without the second check,
        # the pool hands out a zombie connection and the user's first
        # query fails. The peek is one syscall; cheap to run on every
        # acquire.
        if not conn.is_connected or _socket_looks_dead(conn):
            logger.debug(
                "pool.acquire: drain-idle triggered by stale conn=%r closing_idle=%d",
                conn,
                self._pool.qsize(),
            )
            # Close the dead-but-transport-alive connection explicitly:
            # ``_drain_idle`` only walks the idle queue, so a connection
            # we just dequeued whose transport is half-closed (FIN seen)
            # would otherwise leak its writer. ``close()`` is safe on a
            # never-connected conn (``_protocol is None``) — it short-
            # circuits to a noop. Shielded so an outer cancel does not
            # leave the writer dangling.
            #
            # Clear ``_pool_released`` BEFORE close so the close
            # actually runs. ``DqliteConnection.close()`` early-returns
            # when ``_pool_released`` is True (so user-side close on a
            # checked-in conn is a no-op); without this clear, the
            # pool-side close on a dead conn we just dequeued would
            # be silently absorbed and the writer would leak.
            conn._pool_released = False
            try:
                with contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS):
                    await asyncio.shield(conn.close())
            finally:
                # Restore the flag for contract symmetry with
                # ``_drain_idle``: any stale-reference second close()
                # falls through the documented early-return rather
                # than re-running close on a dead conn.
                conn._pool_released = True
            # Wrap _drain_idle so a cancel mid-drain releases the dead
            # conn's reservation. Without this guard, a CancelledError
            # delivered to a checkpoint inside _drain_idle propagates
            # out of acquire() before the _create_connection's except
            # arm runs, leaking one reservation slot per occurrence.
            try:
                await self._drain_idle()
            except BaseException:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())
                raise
            # The dead conn's reservation is re-used for the fresh
            # connection; no counter adjustment needed. (The earlier
            # ``-= 1; += 1`` under the lock was a no-op — kept only
            # because the intent was unclear.) The ``if reserved:``
            # guard the fresh-slot arm uses for double-release safety
            # is not needed here: this arm only runs after a
            # successful ``_pool.get_nowait()`` whose reservation is
            # already held; there is no pre-grant raise to skip past.
            #
            # Clamp the create-connection await to the remaining
            # acquire deadline. Symmetric with the fresh-slot arm
            # above; without the clamp, the cluster.connect retry
            # budget can run for tens of seconds past the user-
            # supplied ``pool.timeout``.
            try:
                create_remaining = deadline - loop.time()
                if create_remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(create_remaining):
                    conn = await self._create_connection()
            except BaseException as exc:
                # Same shielded-release rationale as the new-slot
                # arm above: outer cancel re-arming on the create-
                # connection await must not bypass the decrement.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())
                if isinstance(exc, TimeoutError):
                    idle = self._pool.qsize()
                    checked_out = self._size - idle
                    raise DqliteConnectionError(
                        f"Timed out creating a fresh connection from the pool "
                        f"(pool_id={id(self)}, max_size={self._max_size}, "
                        f"checked_out={checked_out}, idle={idle}, "
                        f"timeout={self._timeout}s)."
                    ) from exc
                raise
            # close() may have run while _create_connection was
            # suspended on leader discovery / TCP handshake. Without
            # this re-check, a fresh connection would be yielded on
            # a pool whose _closed flag is True — contract violation
            # and a sneaky leak (user runs queries against a pool
            # the rest of the program treats as closed).
            if self._closed:
                # Shield both the close and the reservation release so
                # an outer cancel cannot leak the freshly-built
                # connection's transport or the reservation slot.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(conn.close())
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())
                raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
            # Re-check the freshly-dialed conn for definitive death.
            # ``_create_connection`` is supposed to return a fully-
            # established conn with ``is_connected`` True. If
            # ``is_connected`` reports False right after return, the
            # handshake failed silently or the peer tore the conn
            # between completion and return (rare but possible under
            # firewall idle-timeout / peer Raft flip / peer process
            # crash). Yielding such a conn would surface as an opaque
            # transport error on the caller's first query. Catch it
            # at checkout time and raise a clean
            # DqliteConnectionError so caller-side retry has a clear
            # signal. The check is intentionally narrower than the
            # ``_socket_looks_dead`` peek at the top of this arm — the
            # peek false-positives on the standard mock pattern
            # (``MagicMock(spec=DqliteConnection)`` without a mocked
            # ``_protocol``), and the queue-dequeue arm already
            # tolerates dead-looking mocks by going through this same
            # re-create path.
            if not conn.is_connected:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(conn.close())
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())
                raise DqliteConnectionError(
                    "Pool: freshly dialed connection reports is_connected=False; "
                    "handshake failed silently or the peer tore the conn between "
                    f"completion and return (pool_id={id(self)})."
                )

        conn._pool_released = False
        try:
            yield conn
        except BaseException:
            # Cleanup must complete even if a second cancellation lands
            # mid-await. ``returned_to_queue`` tracks whether
            # the reservation was transferred to an in-queue connection;
            # if not, the ``finally`` below releases it. ``asyncio.shield``
            # around ``_release_reservation`` makes the decrement itself
            # uninterruptible so ``_size`` stays consistent under rapid-
            # fire cancellation.
            #
            # Order note: ``conn.close()`` has a guard that returns early
            # when ``_pool_released`` is True (so user code can't
            # accidentally close a connection they returned to the
            # pool). The pool itself MUST therefore close BEFORE setting
            # the flag, or the transport leaks.
            returned_to_queue = False
            try:
                if conn.is_connected and not self._closed:
                    # Connection is healthy — user code raised a non-connection
                    # error. Roll back any open transaction, then return to pool.
                    if await self._reset_connection(conn):
                        # Re-check ``_closed`` after the
                        # ``_reset_connection`` yield: ``pool.close()``
                        # does not take ``_lock`` and may have completed
                        # its drain while we awaited the ROLLBACK.
                        # Putting the conn into a drained queue would
                        # orphan it.
                        if self._closed:
                            try:
                                await conn.close()
                            except _POOL_CLEANUP_EXCEPTIONS:
                                logger.debug(
                                    "pool: ignoring close() error during release",
                                    exc_info=True,
                                )
                            conn._pool_released = True
                        else:
                            try:
                                self._pool.put_nowait(conn)
                            except asyncio.QueueFull:
                                try:
                                    await conn.close()
                                except _POOL_CLEANUP_EXCEPTIONS:
                                    logger.debug(
                                        "pool: ignoring close() error during release",
                                        exc_info=True,
                                    )
                                conn._pool_released = True
                            else:
                                conn._pool_released = True
                                returned_to_queue = True
                    else:
                        try:
                            await conn.close()
                        except _POOL_CLEANUP_EXCEPTIONS:
                            logger.debug(
                                "pool: ignoring close() error during release",
                                exc_info=True,
                            )
                        conn._pool_released = True
                else:
                    # Connection is broken (invalidated by execute/fetch error
                    # handlers). Drain other idle connections — they likely
                    # point to the same dead server.
                    #
                    # Narrow the outer catch to the legitimate transport-
                    # failure set (_POOL_CLEANUP_EXCEPTIONS) so cancellation
                    # and interpreter-exit signals propagate, and any
                    # programmer-bug inside _drain_idle (AttributeError /
                    # TypeError) surfaces instead of being DEBUG-logged into
                    # silence.
                    try:
                        await self._drain_idle()
                    except _POOL_CLEANUP_EXCEPTIONS:
                        logger.debug(
                            "pool.acquire cleanup: _drain_idle failed",
                            exc_info=True,
                        )
                    # Shield ``conn.close()`` so an outer cancel landing
                    # mid-cleanup does not skip it — the user's
                    # checked-out connection's transport would otherwise
                    # stay open until GC. The ``_close_timeout`` bound
                    # on ``wait_closed`` keeps the shielded await
                    # bounded; ``contextlib.suppress(CancelledError)``
                    # absorbs any nested cancel delivered after the
                    # shield released so the original cancel still
                    # propagates via the surrounding ``raise``.
                    # ``_drain_idle()`` above stays bare: it protects
                    # siblings, not the leaked conn, and shielding it
                    # could turn outer cancel into an unbounded wait.
                    try:
                        try:
                            with contextlib.suppress(asyncio.CancelledError):
                                await asyncio.shield(conn.close())
                        except (OSError, DqliteConnectionError):
                            logger.debug(
                                "pool.acquire cleanup: conn.close(%s) failed",
                                sanitize_for_log(str(getattr(conn, "_address", "?"))),
                                exc_info=True,
                            )
                        except RuntimeError:
                            # ``RuntimeError`` from inside the shielded
                            # close — typically ``"Event loop is closed"``
                            # during a racing ``engine.dispose()`` — would
                            # otherwise propagate out and SUPPLANT the
                            # user's original exception (preserved only
                            # via ``__context__``). Log it and absorb so
                            # the bare ``raise`` further out re-raises
                            # the user's original error. Narrow to
                            # ``RuntimeError`` so programming bugs
                            # (``AttributeError``, ``TypeError``, etc.)
                            # still surface.
                            logger.debug(
                                "pool.acquire cleanup: conn.close(%s) raised RuntimeError",
                                sanitize_for_log(str(getattr(conn, "_address", "?"))),
                                exc_info=True,
                            )
                    finally:
                        # Always set ``_pool_released`` so a subsequent
                        # close() short-circuits and the slot accounting
                        # stays consistent — landed even if the shielded
                        # close above raised an unrecognised exception.
                        conn._pool_released = True
            finally:
                if not returned_to_queue:
                    # ``asyncio.shield`` already prevents an outer cancel
                    # from interrupting ``_release_reservation``; the narrow
                    # ``CancelledError`` suppression here only absorbs a
                    # fresh nested cancel delivered between the shielded
                    # return and the end of this block.
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
            raise
        else:
            await self._release(conn)

    async def _reset_connection(self, conn: DqliteConnection) -> bool:
        """Roll back any open transaction before returning to pool.

        Returns True if the connection is clean and can be reused,
        False if it should be destroyed.

        If ROLLBACK raises, the connection's transaction state is
        unknowable from the client's side — the wire request may have
        been half-sent, or delivered but not acknowledged. The pool
        therefore drops the connection; the dqlite cluster's Raft log
        eventually reclaims any uncommitted work from the terminated
        session. A DEBUG log entry is emitted to help operators
        diagnose churning pools.
        """
        # Issue ROLLBACK if any of these is set:
        #   * ``_in_transaction`` — the explicit-tx flag transitioned
        #     via BEGIN / SAVEPOINT-autobegin / transaction().
        #   * ``_savepoint_stack`` — there are tracked SAVEPOINT frames.
        #   * ``_savepoint_implicit_begin`` — the outermost SAVEPOINT
        #     auto-begun the transaction.
        #   * ``_has_untracked_savepoint`` — the tracker observed a
        #     SAVEPOINT verb whose name the parser could not represent
        #     (quoted / backtick / square-bracket / unicode / leading-
        #     digit identifier per the case-sensitivity trade-off in
        #     ``_parse_savepoint_name``). The local stack stayed empty
        #     by design but the server still has the savepoint and an
        #     auto-begun transaction; the pool would otherwise return
        #     a slot with a live tx to the next acquirer.
        # Strict isinstance / type checks defend against unit-test
        # fakes whose attributes are MagicMock instances (truthy by
        # default) — without the type guards, every fake-conn release
        # would unconditionally enter the ROLLBACK branch.
        # Cheap pre-anything liveness check: if the transport is
        # closing or the reader has seen EOF, the connection cannot
        # be re-used regardless of tx state. Without this, a clean-tx
        # but wire-poisoned connection returned to the pool waited
        # in the queue until a future acquirer dequeued it, tripped
        # ``_socket_looks_dead`` on dequeue, and discarded — one
        # wasted slot churn per affected connection. Drop on return
        # so the slot is freed immediately.
        if _socket_looks_dead(conn):
            logger.debug(
                "pool: dropping connection %s (socket looks dead on return)",
                sanitize_for_log(str(conn._address)),
            )
            return False
        in_tx = getattr(conn, "_in_transaction", False)
        sp_stack = getattr(conn, "_savepoint_stack", None)
        implicit_begin = getattr(conn, "_savepoint_implicit_begin", False)
        untracked_sp = getattr(conn, "_has_untracked_savepoint", False)
        needs_rollback = (
            (isinstance(in_tx, bool) and in_tx)
            or (isinstance(sp_stack, list) and bool(sp_stack))
            or (isinstance(implicit_begin, bool) and implicit_begin)
            or (isinstance(untracked_sp, bool) and untracked_sp)
        )
        if needs_rollback:
            try:
                await conn.execute(_TRANSACTION_ROLLBACK_SQL)
            except _POOL_CLEANUP_EXCEPTIONS as exc:
                # Distinguish "server already auto-rolled back" (the
                # deterministic SQLITE_ERROR + "no transaction is
                # active" reply) from a real ROLLBACK failure. The
                # benign case happens during a leader-flip cascade:
                # the new leader has no record of our tx, so the
                # ROLLBACK ack reports tx-none. The connection itself
                # is healthy — preserve the slot, scrub the local
                # flags, and let the next acquirer reuse it instead
                # of paying a fresh-connect round-trip per slot.
                # ``transaction()`` applies the same discrimination
                # at its rollback site.
                if _is_no_tx_rollback_error(exc):
                    logger.debug(
                        "pool: ROLLBACK on %s found no active transaction "
                        "(server-side tx already gone); preserving connection",
                        sanitize_for_log(str(conn._address)),
                    )
                    conn._in_transaction = False
                    conn._tx_owner = None
                    if hasattr(conn, "_savepoint_stack"):
                        conn._savepoint_stack.clear()
                    if hasattr(conn, "_savepoint_implicit_begin"):
                        conn._savepoint_implicit_begin = False
                    if hasattr(conn, "_has_untracked_savepoint"):
                        conn._has_untracked_savepoint = False
                    return True
                # Differentiate leader-flip (normal cluster churn —
                # DEBUG) from genuine server failure (latent bug or
                # server fault — WARNING with traceback).
                code = getattr(exc, "code", None)
                if code in LEADER_ERROR_CODES:
                    logger.debug(
                        "pool: dropping connection %s after leader-class "
                        "ROLLBACK failure (code=%s)",
                        sanitize_for_log(str(conn._address)),
                        code,
                    )
                else:
                    # Format string uses %s for both substitutions
                    # because the values are pre-stringified — the
                    # exception is sanitised via repr() so the
                    # OperationalError(...) shape reaches operators
                    # without %r doubly-quoting an already-sanitised
                    # string. Aligns with the round-33
                    # ``pool.initialize`` per-failure-warning shape.
                    logger.warning(
                        "pool: dropping connection %s after ROLLBACK failure: %s",
                        sanitize_for_log(str(conn._address)),
                        sanitize_for_log(repr(exc)),
                        exc_info=True,
                    )
                return False
            conn._in_transaction = False
            conn._tx_owner = None
            if hasattr(conn, "_savepoint_stack"):
                conn._savepoint_stack.clear()
            if hasattr(conn, "_savepoint_implicit_begin"):
                conn._savepoint_implicit_begin = False
            if hasattr(conn, "_has_untracked_savepoint"):
                conn._has_untracked_savepoint = False
        return True

    async def _release(self, conn: DqliteConnection) -> None:
        """Return a connection to the pool or close it.

        ``conn.close()`` has an early-return guard against
        ``_pool_released=True``, so close MUST run before the flag is
        set — otherwise the transport leaks (a bug that affects every
        branch that closes a pool-owned connection).

        The reservation release lives in an outer ``finally`` so it
        runs even when ``CancelledError`` is delivered during the
        preceding ``_reset_connection`` (the ROLLBACK await) or the
        inline ``conn.close()`` calls — a per-branch shield could only
        protect the decrement once reached, not the I/O that has to
        complete first. Symmetric with the exception-path
        ``returned_to_queue`` flag in ``acquire()``.
        """
        # Fork-after-acquire: the ``async with pool.acquire():`` block
        # straddled a ``fork()``; the implicit ``__aexit__`` is now
        # running in the child. Touching the parent-loop-bound
        # ``self._lock`` (an ``asyncio.Lock``) from the child raises
        # ``RuntimeError("got Future <Future pending> attached to a
        # different loop")`` from asyncio internals, masking the
        # canonical ``InterfaceError("Pool used after fork...")``
        # diagnostic that ``acquire`` and ``close`` produce. Mark the
        # conn released so its finalizer doesn't surface a spurious
        # ``ResourceWarning`` at GC; the pool's accounting stays as it
        # was in the parent, which is the correct snapshot for the
        # child (it never holds the slot — it only inherited the
        # reference). Symmetric with ``acquire``'s and ``close``'s
        # fork short-circuits.
        if _conn_mod._current_pid != self._creator_pid:
            with contextlib.suppress(AttributeError):
                conn._pool_released = True
            return
        returned_to_queue = False
        try:
            if self._closed:
                with contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS):
                    await conn.close()
                conn._pool_released = True
                return

            if not await self._reset_connection(conn):
                with contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS):
                    await conn.close()
                conn._pool_released = True
                return

            # Re-check ``_closed`` after the ``_reset_connection`` yield:
            # ``pool.close()`` does not take ``_lock``, so it may have
            # set ``_closed=True`` and drained the queue while we were
            # awaiting the ROLLBACK. Putting the conn into a drained
            # queue would orphan it (acquire() raises Pool is closed,
            # the conn is unreachable). Mirrors the close-vs-initialize
            # symmetry fix.
            if self._closed:
                with contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS):
                    await conn.close()
                conn._pool_released = True
                return

            # Healthy connection returning to queue: no close; just flip
            # the flag and enqueue. If the queue is full we must close
            # before setting the flag.
            #
            # Synchronous-tail invariant: there is no ``await`` between
            # the post-reset ``self._closed`` re-check above and the
            # ``put_nowait`` below, so under standard CPython
            # single-loop semantics no concurrent ``pool.close()`` can
            # interleave to drain the queue between the check and the
            # put. Do NOT introduce an ``await`` in this block without
            # adding a fresh ``_closed`` re-check immediately before
            # ``put_nowait`` — that would re-open a TOCTOU window where
            # a concurrent ``close()`` finishes its drain after our
            # check but before our enqueue, orphaning the released
            # connection (no longer reachable from the pool, never
            # closed).
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                with contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS):
                    await conn.close()
                conn._pool_released = True
            else:
                conn._pool_released = True
                # Reservation transfers to the queued connection's
                # lifetime; do not decrement.
                returned_to_queue = True
        finally:
            if not returned_to_queue:
                # On CancelledError during ``_reset_connection`` (the
                # ROLLBACK await) or during one of the inline
                # ``conn.close()`` awaits above, we fall into this
                # branch with ``_pool_released`` still False. The
                # client-side ``_run_protocol`` already invalidated
                # the transport on cancel, so the socket is not
                # leaking — but the DqliteConnection bookkeeping
                # expects ``_pool_released=True`` for every conn that
                # has passed through ``_release``. Mark it here so a
                # stale reference later observing ``conn.close()``
                # takes the early-return path instead of running a
                # redundant close against a protocol that's already
                # None.
                #
                # Drain ``_pending_drain`` BEFORE setting
                # ``_pool_released=True``: a cancel mid-ROLLBACK has
                # ``_invalidate`` schedule a bounded ``wait_closed``
                # drain task on the connection. ``close()`` would
                # normally await that task at ``connection.py``'s
                # pending-drain block, but the early-return on
                # ``_pool_released=True`` we set below short-circuits
                # past it. Snapshot and shield-await the drain here so
                # the reader-task doesn't outlive the connection (no
                # "Task was destroyed but it is pending" on shutdown).
                pending = getattr(conn, "_pending_drain", None)
                if pending is not None and not pending.done():
                    # Absorb the await-side CancelledError so a fresh
                    # outer cancel during release does not tear down
                    # the bounded drain (the shield itself protects
                    # the inner task; this suppress covers the
                    # awaiter-side raise delivered when the
                    # surrounding scope is cancelled). Do NOT suppress
                    # KeyboardInterrupt / SystemExit — those must
                    # propagate.
                    #
                    # The Exception arm is belt-and-braces: production
                    # drains run inside ``_bounded_drain`` which
                    # already wraps in ``suppress(Exception)``. Log
                    # the suppressed exception at DEBUG so a future
                    # drain bug isn't completely invisible — sibling
                    # discipline to ``_drain_idle``'s per-iteration
                    # ``logger.debug("close failed", exc_info=True)``.
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        # Sanitise the address through the wire helper
                        # for symmetry with the other pool log sites —
                        # connection addresses originate from caller
                        # config but a hostile resolver / forged host
                        # entry can still smuggle control characters
                        # via DNS canonicalisation.
                        logger.debug(
                            "pool _release: suppressed pending-drain exception on conn %s",
                            sanitize_for_log(str(getattr(conn, "_address", "?"))),
                            exc_info=True,
                        )
                conn._pool_released = True
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.execute(sql, params)

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.fetch(sql, params)

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute a query and return the first result using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.fetchone(sql, params)

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        """Execute a query and return results as lists using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.fetchall(sql, params)

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Execute a query and return a single value using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.fetchval(sql, params)

    async def close(self) -> None:
        """Close the pool and all idle connections.

        Sets the pool as closed so no new connections can be acquired.
        Idle connections are closed immediately. In-use connections are
        closed when they are returned (when the acquire() context manager
        exits). To ensure all connections are closed, cancel or await
        in-flight tasks before calling close().

        **No drain-completion guarantee for in-flight checked-out
        connections.** ``close()`` returns when the *idle* queue has
        been drained, NOT when checked-out connections have been
        returned. After ``close()`` returns, the reservation count
        (``self._size``) may still be non-zero — the still-checked-out
        connections close on return via ``_release``'s `_closed` branch,
        but the caller of ``close()`` does not block on that. A second
        ``close()`` call after the first completes also returns
        immediately via ``_close_done`` regardless of whether
        in-flight returns have completed. Operators expecting
        ``engine.dispose()``-style "all connections closed" semantics
        must cancel or await in-flight tasks first; there is no
        ``wait_drained()`` API by design (matches go-dqlite's pool
        which uses Go's ``database/sql.DB.Close`` close-on-return
        model).

        Idempotent and concurrent-caller-safe: a second caller (or a
        re-entry after completion) waits on the first caller's drain
        via ``_close_done`` rather than racing ``_drain_idle``.
        """
        # Fork-after-init: the inherited connection FDs are shared with
        # the parent. Draining and writer.close() in the child would
        # send FIN on sockets the parent still uses. Flip the closed
        # flag so the child's references can be GC'd quietly without
        # touching the wire. The child cannot acquire new connections
        # either way (pid-aware ``acquire`` rejects). Symmetric with
        # ``DqliteConnection.close``'s fork short-circuit.
        #
        # The fork-pid check runs BEFORE the ``_closed`` early-return
        # so a child whose parent forked while mid-close (with
        # ``_close_done`` set but not yet ``set()``) does not block on
        # an Event bound to the parent's loop. Awaiting that Event in
        # the child's fresh loop hangs forever.
        if _conn_mod._current_pid != self._creator_pid:
            self._closed = True
            self._closed_flag[0] = True
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            return
        if self._closed:
            if self._close_done is not None:
                await self._close_done.wait()
            # If the FIRST caller's drain was interrupted mid-flight
            # by an outer cancel (``asyncio.timeout(pool.close())``
            # under SIGTERM-with-budget), ``_close_done`` was set in
            # the finally even though the queue is still under-
            # drained. Run a best-effort sweep so the second caller's
            # ``await pool.close()`` returns against an empty queue
            # rather than the documented "drain completed" contract
            # being silently broken.
            if not self._drain_complete:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._drain_remaining_after_cancel())
            return
        # Publish the drain-done event BEFORE flipping the closed flag
        # so any second caller observing ``_closed=True`` is guaranteed
        # to see a valid ``_close_done`` to wait on. Under single-task
        # asyncio this ordering is invisible; under signal-handler
        # delivery (KeyboardInterrupt landing on a bytecode check
        # between two assignments) it closes the window where a second
        # caller saw ``_closed=True`` with ``_close_done is None`` and
        # short-circuited without waiting on the first caller's drain.
        self._close_done = asyncio.Event()
        self._closed = True
        self._closed_flag[0] = True
        try:
            # Move the finalizer-detach + drain inside the try so the
            # finally's ``_close_done.set()`` is reachable even if a
            # ``BaseException`` (KeyboardInterrupt / SystemExit / a
            # synthetic bytecode-tight signal) lands between flag
            # publication and the first awaited line. Without this,
            # a second caller in the early-return arm at lines
            # 1815-1817 awaits ``_close_done`` forever.
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            logger.debug(
                "pool.close: draining idle=%d in_flight=%d",
                self._pool.qsize(),
                max(self._size - self._pool.qsize(), 0),
            )
            if self._closed_event is not None:
                self._closed_event.set()
            await self._drain_idle()
            # Drain completed normally. Set BEFORE the finally so a
            # cancel landing between the drain return and the flag
            # assignment leaves ``_drain_complete=False`` — siblings
            # then run the best-effort sweep above.
            self._drain_complete = True
        finally:
            self._close_done.set()
            # Drop the signalled-and-now-useless wakeup event
            # so it can be GC'd alongside the pool's other
            # once-used loop-bound primitives (``_async_conn``,
            # ``_protocol``, ``_connect_lock``, ``_op_lock``,
            # ``_loop_ref`` are all nulled at the end of their
            # close paths). ``_close_done`` stays — the
            # second-caller arm above awaits it; clearing in
            # the second-caller arm would be TOCTOU and clearing
            # here while siblings might still be parked is wrong.
            # ``_closed_event`` has no remaining waiters once
            # ``set()`` runs (the pool is closed; new acquirers
            # short-circuit before parking) so it is safe to drop.
            self._closed_event = None

        # In-use connections are closed by acquire()'s cleanup when they
        # return — the else branch checks _closed and closes instead of
        # returning to the pool. Force-closing them here would race with
        # the acquire context manager and corrupt _size.

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
