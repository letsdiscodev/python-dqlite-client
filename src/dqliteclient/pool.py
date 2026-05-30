"""Connection pooling for dqlite."""

import asyncio
import contextlib
import logging
import math
import os
import sys
import warnings
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Final, NoReturn, Self

from dqliteclient import connection as _conn_mod
from dqliteclient._dial import DialFunc
from dqliteclient.cluster import ClusterClient, RedirectPolicy, _observe_drain_exception
from dqliteclient.connection import (
    _TRANSACTION_ROLLBACK_SQL,
    CLOSE_TIMEOUT_FLOOR,
    CLOSE_TIMEOUT_FLOOR_RATIONALE,
    DEFAULT_CLOSE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
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
from dqliteclient.protocol import _is_int_not_bool, validate_positive_int_or_none
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)
from dqlitewire import LEADER_ERROR_CODES, sanitize_for_log

__all__ = ["ConnectionPool"]

# Exceptions a best-effort pool cleanup (ROLLBACK, close()) may raise on a
# partially-torn-down transport; everything else (cancels, KeyboardInterrupt,
# programming errors) must propagate. InterfaceError is included so a ROLLBACK
# that hits "owned by another task" drops the conn instead of leaking a slot.
_POOL_CLEANUP_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    OSError,
    DqliteConnectionError,
    ProtocolError,
    OperationalError,
    InterfaceError,
)

# Per-connection drain cap multiplier: _close_impl worst-case is
# ``(_CLOSE_RESNAPSHOT_CAP + 1) × close_timeout``. Derived from the connection
# module's constant so a bump to the cap propagates without manual edits.
_DRAIN_PER_CONN_CAP_MULTIPLIER: Final[int] = _conn_mod._CLOSE_RESNAPSHOT_CAP + 1

logger = logging.getLogger(__name__)


def _socket_looks_dead(conn: DqliteConnection) -> bool:
    """Best-effort local detection of a half-closed TCP socket.

    True if _protocol is missing/None, wire-incoherent, transport closing, or
    reader at EOF. Peek failures on partial mocks count as alive. Must run on
    the loop thread with no await between peeks (no torn reads vs _invalidate).
    """
    protocol = getattr(conn, "_protocol", None)
    if protocol is None:
        return True
    # Poisoned-decoder short-circuit: wire desync means a ROLLBACK round-trip
    # is wasted RTT before ProtocolError → invalidate, so drop now.
    try:
        coherent = protocol.is_wire_coherent
    except (AttributeError, RuntimeError):
        coherent = True
    if isinstance(coherent, bool) and not coherent:
        return True
    writer = getattr(protocol, "_writer", None)
    reader = getattr(protocol, "_reader", None)
    transport = getattr(writer, "transport", None) if writer is not None else None
    # Narrow suppression so a refactor's misnamed-attribute bug still surfaces.
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
    *,
    # Capture stdlib module globals so a finalize callback firing after
    # Py_FinalizeEx phase-3 nulls them does not raise into the GC machinery.
    # get_current_pid is deliberately NOT captured: fork-pid tests patch the
    # module-level name, and a default-capture would freeze it past the patch.
    _warnings: Any = warnings,
    _contextlib: Any = contextlib,
) -> None:
    """Emit a ResourceWarning when a ConnectionPool is GC'd without close().

    Gated three ways: skip in a forked child (parent owns the lifecycle), skip
    if close() ran, skip if no slot was ever reserved. ``queue`` lets the
    message report the queued-conn count so the N+1 per-conn warnings correlate.
    """
    if _warnings is None or _contextlib is None:
        return
    if creator_pid is not None:
        # Read get_current_pid from module globals at call time so the test
        # patch is observed; try absorbs the shutdown-time TypeError.
        try:
            if get_current_pid() != creator_pid:
                return
        except Exception:
            return
    if closed_flag[0] or not reserved_flag[0]:
        return
    queued_count: int | None = None
    if queue is not None:
        with _contextlib.suppress(Exception):
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
    with _contextlib.suppress(RuntimeError):
        _warnings.warn(
            message,
            ResourceWarning,
            stacklevel=2,
        )


class ConnectionPool:
    """Connection pool with automatic leader detection.

    NOT thread-safe: use from a single event loop; cross-thread work must go
    through ``asyncio.run_coroutine_threadsafe()``. Free-threaded Python (no-GIL)
    is not supported.
    """

    def __init__(
        self,
        addresses: Sequence[str] | None = None,
        *,
        database: str = "default",
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        cluster: ClusterClient | None = None,
        node_store: NodeStore | None = None,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        max_message_size: int | None = None,
        max_attempts: int | None = None,
        max_elapsed_seconds: float | None = None,
        dial_func: DialFunc | None = None,
        concurrent_leader_conns: int | None = None,
        redirect_policy: RedirectPolicy | None = None,
    ) -> None:
        """Initialize a connection pool (does not connect until ``initialize()``).

        ``addresses`` is ignored when ``cluster`` or ``node_store`` is given.
        ``cluster`` (and the ``node_store`` it is built from) is caller-owned — the
        pool does not close it — and lets multiple pools share one cluster;
        ``cluster``/``node_store`` are mutually exclusive, as are ``cluster`` and
        ``dial_func`` (the cluster carries its own). ``min_size`` connections are
        pre-warmed at ``initialize()`` but are NOT a steady-state floor — the pool
        does not replenish after a drain. ``timeout`` is per-RPC-phase (each phase
        gets the full budget, so a call can take ~N × ``timeout``) and is the
        default for ``dial_timeout``/``attempt_timeout`` and the per-acquire
        wall-clock clamp. ``max_total_rows`` (``None`` disables, dropping the
        slow-drip memory bound — avoid in production) and ``max_continuation_frames``
        bound decode work against a server sending one row per frame.
        ``trust_server_heartbeat`` widens the per-read deadline to the
        server-advertised heartbeat (300 s cap). ``close()`` drains serially, so
        total close wall-clock ~ qsize × ``close_timeout``; size it against SIGTERM
        grace periods. ``max_attempts`` (``None`` uses the cluster default of 3,
        must be >= 1) and ``max_elapsed_seconds`` (``None`` lets only
        ``max_attempts`` govern) bound leader discovery.
        """
        # Reject bool first: True/False coerce to valid int sizes and would mask
        # a caller accidentally passing a flag.
        if not _is_int_not_bool(min_size):
            raise TypeError(f"min_size must be int, got {type(min_size).__name__}")
        if not _is_int_not_bool(max_size):
            raise TypeError(f"max_size must be int, got {type(max_size).__name__}")
        if max_attempts is not None and not _is_int_not_bool(max_attempts):
            raise TypeError(f"max_attempts must be int or None, got {type(max_attempts).__name__}")
        if min_size < 0:
            raise ValueError(f"min_size must be non-negative, got {min_size}")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) must not exceed max_size ({max_size})")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1 if provided, got {max_attempts}")
        # Wording mirrors retry.py exactly so caller-side message parsing stays
        # consistent across layers.
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
            min_value=CLOSE_TIMEOUT_FLOOR,
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
        # cluster= carries its own dial_func; supplying both would silently
        # desync the pool's intended dialer from the cluster's actual one.
        if cluster is not None and dial_func is not None:
            raise ValueError(
                "dial_func cannot be combined with cluster=; "
                "configure dial_func on the externally-owned ClusterClient"
            )
        # Same mutual-exclusion rationale as dial_func: cluster= carries its own.
        if cluster is not None and concurrent_leader_conns is not None:
            raise ValueError(
                "concurrent_leader_conns cannot be combined with cluster=; "
                "configure concurrent_leader_conns on the externally-owned ClusterClient"
            )
        if cluster is not None and redirect_policy is not None:
            raise ValueError(
                "redirect_policy cannot be combined with cluster=; "
                "configure redirect_policy on the externally-owned ClusterClient"
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
        # None defers to the wire-layer default (64 MiB); validated downstream.
        self._max_message_size = max_message_size
        self._trust_server_heartbeat = trust_server_heartbeat
        self._close_timeout = close_timeout
        self._max_attempts = max_attempts
        self._max_elapsed_seconds = max_elapsed_seconds

        if cluster is not None:
            # Caller-supplied; pool does not override its timeouts.
            self._cluster = cluster
        elif node_store is not None:
            cluster_kwargs: dict[str, object] = dict(
                timeout=timeout,
                dial_timeout=dial_timeout,
                attempt_timeout=attempt_timeout,
                max_total_rows=max_total_rows,
                max_continuation_frames=max_continuation_frames,
                max_message_size=max_message_size,
                trust_server_heartbeat=trust_server_heartbeat,
                dial_func=dial_func,
            )
            if concurrent_leader_conns is not None:
                cluster_kwargs["concurrent_leader_conns"] = concurrent_leader_conns
            if redirect_policy is not None:
                cluster_kwargs["redirect_policy"] = redirect_policy
            self._cluster = ClusterClient(node_store, **cluster_kwargs)  # type: ignore[arg-type]
        else:
            cluster_kwargs = dict(
                timeout=timeout,
                dial_timeout=dial_timeout,
                attempt_timeout=attempt_timeout,
                max_total_rows=max_total_rows,
                max_continuation_frames=max_continuation_frames,
                max_message_size=max_message_size,
                trust_server_heartbeat=trust_server_heartbeat,
                dial_func=dial_func,
            )
            if concurrent_leader_conns is not None:
                cluster_kwargs["concurrent_leader_conns"] = concurrent_leader_conns
            if redirect_policy is not None:
                cluster_kwargs["redirect_policy"] = redirect_policy
            self._cluster = ClusterClient.from_addresses(self._addresses, **cluster_kwargs)  # type: ignore[arg-type]
        self._pool: asyncio.Queue[DqliteConnection] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        # Loop-binding guard: asyncio.Queue/Lock lazy-bind to a loop on first
        # use; lazy-bound here too so factory construction off-loop still works.
        self._loop_ref: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self._closed = False
        self._closed_event: asyncio.Event | None = None
        self._close_done: asyncio.Event | None = None
        # Strong-ref set keeps fire-and-forget _release_after_drain follow-ups
        # rooted so GC cannot reclaim one mid-flight (leaking a reservation slot)
        # during loop teardown; the done-callback discards each entry.
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Distinguishes "first caller left the drain phase" (signalled via
        # _close_done so siblings don't deadlock under cancel) from "drain ran to
        # completion" — a second caller re-checks this and sweeps if needed.
        self._drain_complete: bool = False
        self._initialized = False
        # 3-phase initialize coordination; both fields nulled by the close-side
        # fork shortcut so a child initialize() doesn't park on a parent's Event.
        self._initializing: bool = False
        self._initialize_done_event: asyncio.Event | None = None
        # Fork-after-init is unsupported (shared sockets + parent-loop
        # primitives); store creator pid so cross-fork acquire raises cleanly.
        self._creator_pid = os.getpid()
        # ResourceWarning finalizer. Mutable cells so the finalizer reads current
        # values; _reserved_flag flips True on first slot reservation so a
        # never-acquired pool stays silent at GC.
        self._closed_flag: list[bool] = [False]
        self._reserved_flag: list[bool] = [False]
        # Pass the queue so the warning reports the queued-conn count; the queue
        # holds no back-reference to the pool, so no cycle is introduced.
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
        """Whether ``close()`` has been called on this pool."""
        return self._closed

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        # Show the first few addresses with a count hint so the repr stays short.
        addrs = self._addresses[:3]
        suffix = f"+{len(self._addresses) - 3}" if len(self._addresses) > 3 else ""
        addrs_repr = f"{addrs!r}{suffix}"
        return (
            f"ConnectionPool(addresses={addrs_repr}, "
            f"size={self._size}, min_size={self._min_size}, "
            f"max_size={self._max_size}, {state})"
        )

    def __reduce__(self) -> "NoReturn":
        # asyncio.Queue/Lock pickle on 3.10+, so a naive pickle.dumps(pool)
        # silently yields a loop-detached duplicate that corrupts on use.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — holds "
            f"loop-bound asyncio.Queue / asyncio.Lock / asyncio.Event "
            f"and live worker tasks; reconstruct from configuration "
            f"in the target process instead."
        )

    def _check_loop_binding(self) -> None:
        """Validate the pool's asyncio primitives match the current loop.

        Lazy-bind on first call, verify on every subsequent call: cross-loop
        misuse otherwise surfaces only as a deep asyncio-internal error.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise InterfaceError("ConnectionPool must be used from within an async context.") from e
        # getattr-safe for __new__-built test fixtures that bypass __init__.
        loop_ref = getattr(self, "_loop_ref", None)
        if loop_ref is None:
            self._loop_ref = weakref.ref(current_loop)
            return
        bound = loop_ref()
        if bound is None:
            raise InterfaceError(
                "ConnectionPool is bound to a closed event loop. "
                "Reconstruct the pool in the new loop."
            )
        if current_loop is not bound:
            raise InterfaceError(
                "ConnectionPool is bound to a different event loop. "
                "Do not share pools across event loops or OS threads."
            )

    async def initialize(self) -> None:
        """Initialize the pool with ``min_size`` connections.

        Idempotent: concurrent callers share one initialization. Warm-up dials
        run in parallel but every connect targets the current leader, so a large
        min_size does not speed up startup — keep it in the single digits.
        """
        if _conn_mod.get_current_pid() != self._creator_pid:
            raise InterfaceError(
                f"Pool used after fork; reconstruct from configuration "
                f"in the target process. (created in pid {self._creator_pid}, "
                f"current pid {_conn_mod.get_current_pid()})"
            )
        self._check_loop_binding()
        # 3-phase coordination: Phase A reserves under _lock (no awaits), Phase B
        # gathers the dials lock-free so concurrent acquire() can progress, Phase
        # C re-takes _lock to publish. Mirrors psycopg_pool's open()/wait().
        async with self._lock:
            if self._initialized:
                return
            if self._initializing:
                # Secondary caller: snapshot the event under the lock so Phase C's
                # null-out can't race us; await it outside the lock.
                evt = self._initialize_done_event
            else:
                if self._closed:
                    raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
                if self._min_size <= 0:
                    self._initialized = True
                    return
                self._initializing = True
                self._initialize_done_event = asyncio.Event()
                self._size += self._min_size
                if self._size > 0:
                    self._reserved_flag[0] = True
                evt = None
        if evt is not None:
            # Park outside the lock so Phase B is not blocked, then recurse to
            # honour idempotence (the first caller may have failed and a retry is
            # allowed). Bounded: each iteration exits or wins the Phase A race.
            await evt.wait()
            return await self.initialize()
        logger.debug("pool.initialize: requesting %d connections", self._min_size)
        # Reservations still needing release: the failure-finally releases
        # exactly ``unqueued`` slots (those never published to the queue).
        unqueued = self._min_size
        unqueued_survivors: list[DqliteConnection] = []
        # Build tasks explicitly (not the gather-* form) and INSIDE the try so a
        # CancelledError out of gather, or a BaseException mid-construction, still
        # leaves the finally with a handle on every child to close its conn.
        create_tasks: list[asyncio.Task[DqliteConnection]] = []
        # gather raising CancelledError skips the post-gather code, so the finally
        # walks create_tasks to recover completed children's results.
        gather_returned = False
        # Whether Phase B reached "all succeeded" — gates the failure-finally.
        success_branch = False
        successes: list[DqliteConnection] = []
        try:
            for _ in range(self._min_size):
                create_tasks.append(asyncio.create_task(self._create_connection()))
            results = await asyncio.gather(
                *create_tasks,
                return_exceptions=True,
            )
            gather_returned = True
            failures: list[BaseException] = []
            for r in results:
                if isinstance(r, BaseException):
                    failures.append(r)
                else:
                    successes.append(r)
            if failures:
                # Sanitise str(exc), not repr(exc): repr would double-encode the
                # control chars; prepend the type name to keep the class cue.
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
                # Log each failure individually: partial outages produce
                # different root causes per node that a single surface would hide.
                for i, exc in enumerate(failures):
                    logger.warning(
                        "pool.initialize: create_connection %d/%d failed: %s: %s",
                        i + 1,
                        len(failures),
                        type(exc).__name__,
                        sanitize_for_log(str(exc)),
                    )
                await self._initialize_close_unqueued(successes)
                # Single failure keeps its narrow type so ``except
                # DqliteConnectionError`` still matches; multiple raise a group.
                if len(failures) == 1:
                    raise failures[0]
                # Bounded so the exception graph stays picklable for cross-process
                # capture (Celery, multiprocessing).
                from dqliteclient.cluster import _bounded_group

                raise _bounded_group(
                    f"pool.initialize: {len(failures)} of {self._min_size} connects failed",
                    failures,
                )
            # All succeeded — hand to Phase C; record on unqueued_survivors so a
            # cancel before the lock acquire still routes them to the close-walk.
            success_branch = True
            unqueued_survivors = list(successes)
        finally:
            # Capture the in-flight exception so a later raise can chain it via
            # __cause__ (SA's walk_cause_chain follows __cause__, not __context__).
            in_flight: BaseException | None = sys.exc_info()[1]
            # Cancel-during-gather recovery: completed children's results sit in
            # their tasks; walk create_tasks to recover them since the post-gather
            # assignment never ran.
            if not gather_returned:
                cancelled_pending: list[asyncio.Task[DqliteConnection]] = []
                for t in create_tasks:
                    if not t.done():
                        t.cancel()
                        cancelled_pending.append(t)
                outer_cancel: BaseException | None = None
                if cancelled_pending:
                    try:
                        await asyncio.gather(*cancelled_pending, return_exceptions=True)
                    except BaseException as exc:
                        outer_cancel = exc
                self._initialize_collect_completed_conns(create_tasks, unqueued_survivors)
                if outer_cancel is not None:
                    # Close, release, and clear the Phase A flags before re-raising
                    # so a subsequent initialize() isn't blocked on stale state.
                    await self._initialize_close_unqueued(unqueued_survivors)
                    async with self._lock:
                        if unqueued > 0 and self._release_reservations_locked(unqueued):
                            self._signal_state_change()
                        self._initializing = False
                        if self._initialize_done_event is not None:
                            self._initialize_done_event.set()
                            self._initialize_done_event = None
                    if in_flight is not None and in_flight is not outer_cancel:
                        raise outer_cancel from in_flight
                    raise outer_cancel
            # Phase B failure tail: close survivors, release reservations, and
            # clear the Phase A flags so secondary callers wake and can retry.
            if not success_branch:
                async with self._lock:
                    if unqueued > 0 and self._release_reservations_locked(unqueued):
                        self._signal_state_change()
                    self._initializing = False
                    if self._initialize_done_event is not None:
                        self._initialize_done_event.set()
                        self._initialize_done_event = None
                await self._initialize_close_unqueued(unqueued_survivors)
        # Phase C — publish under lock, NO await inside. put_nowait cannot raise
        # QueueFull in correct state (maxsize=max_size >= min_size); the except
        # arm is defensive for test monkey-patches. If close() ran during Phase B,
        # route survivors to the close helper instead of publishing.
        close_these: list[DqliteConnection] = []
        publish_error: BaseException | None = None
        async with self._lock:
            try:
                if self._closed:
                    close_these = list(successes)
                    if self._release_reservations_locked(self._min_size):
                        self._signal_state_change()
                else:
                    for conn in successes:
                        self._pool.put_nowait(conn)
                    self._initialized = True
            except Exception as exc:
                publish_error = exc
                close_these = list(successes)
                if self._release_reservations_locked(self._min_size):
                    self._signal_state_change()
            self._initializing = False
            if self._initialize_done_event is not None:
                self._initialize_done_event.set()
                self._initialize_done_event = None
        if close_these:
            await self._initialize_close_unqueued(close_these)
        if publish_error is not None:
            raise publish_error
        if not close_these:
            logger.debug("pool.initialize: %d connections ready", self._min_size)

    @staticmethod
    def _initialize_collect_completed_conns(
        create_tasks: list[asyncio.Task[DqliteConnection]],
        unqueued_survivors: list[DqliteConnection],
    ) -> None:
        """Append conns from completed-and-not-cancelled create_tasks.

        Shared by both initialize recovery arms so they stay in lockstep.
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

        Each close is shielded so an outer cancel mid-walk doesn't orphan the
        rest; _POOL_CLEANUP_EXCEPTIONS land in DEBUG, anything else propagates.
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
                # Close completes via the shield; the cancel is intentionally not
                # re-raised — survivor cleanup is best-effort.
                logger.debug(
                    "pool.initialize: cancel during unqueued-survivor close",
                    exc_info=True,
                )

    async def _close_best_effort(self, conn: DqliteConnection, site: str) -> None:
        """Shielded close that logs _POOL_CLEANUP_EXCEPTIONS at DEBUG.

        ``site`` tags the log key (``"pool.<site>: close error"``). CancelledError
        is intentionally NOT caught — the shield runs the close to completion but
        the cancel still propagates to the caller.
        """
        # Explicit Task + _observe_drain_exception so a cancel of the outer await
        # doesn't leave the implicit task unobserved ("Task exception was never
        # retrieved" at GC). Same pattern as _drain_idle and _abort_protocol.
        close_task = asyncio.ensure_future(conn.close())
        close_task.add_done_callback(_observe_drain_exception)
        try:
            await asyncio.shield(close_task)
        except _POOL_CLEANUP_EXCEPTIONS:
            logger.debug("pool.%s: close error", site, exc_info=True)

    async def _create_connection(self) -> DqliteConnection:
        """Create a new connection to the leader.

        Does NOT mutate ``_size``: callers must reserve a slot under ``_lock``
        first and release it on failure.
        """
        return await self._cluster.connect(
            database=self._database,
            max_total_rows=self._max_total_rows,
            max_continuation_frames=self._max_continuation_frames,
            trust_server_heartbeat=self._trust_server_heartbeat,
            close_timeout=self._close_timeout,
            max_attempts=self._max_attempts,
            max_elapsed_seconds=self._max_elapsed_seconds,
            max_message_size=self._max_message_size,
        )

    async def _put_back_or_release_late_winner(self, conn: DqliteConnection) -> None:
        """Put a connection back on the queue, or close + release if full.

        Handles the acquire() races where an outer cancel/timeout snapshot beat a
        winning ``get_task``: without this routing the conn is referenced only by
        the soon-GC'd task and its reservation slot leaks permanently.
        """
        # If close() ran since the snapshot, put_nowait would park the conn in a
        # queue no one revisits (drain runs once per close); close+release instead.
        if self._closed:
            # Clear _pool_released BEFORE close or close() early-returns and the
            # transport leaks; restore after for contract symmetry with _drain_idle.
            conn._pool_released = False
            try:
                await self._close_best_effort(conn, "acquire-late-winner-closed")
            finally:
                conn._pool_released = True
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._release_reservation())
            return
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Reservations track queue capacity exactly, so QueueFull is
            # "impossible"; close + adjust the count so the pool shrinks cleanly
            # rather than leaking the reader task and socket.
            conn._pool_released = False
            try:
                await self._close_best_effort(conn, "acquire-late-winner-queuefull")
            finally:
                conn._pool_released = True
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._release_reservation())

    def _release_reservations_locked(self, n: int) -> bool:
        """Decrement ``_size`` by ``n`` (>= 1) with the under-flow guard.

        Caller must already hold ``_lock``. Returns True if the decrement
        happened (caller should signal state-change), False if the guard refused
        an under-flow. The refusal path also signals state-change defensively so a
        waiter re-evaluates against the current ``_size``.

        n<0 would silently increment _size (the guard fails open), so n>=1 (int,
        not bool) is required. The lock-held check is runtime (not assert, which
        -O strips) and best-effort: locked() can't tell which task holds it.
        """
        if not self._lock.locked():
            raise AssertionError(
                "_release_reservations_locked called without _lock held by anyone "
                "(asyncio.Lock has no owner tracking; cannot detect cross-task "
                "contract violation — callers must use 'async with self._lock' "
                "in the same coroutine frame)"
            )
        if not _is_int_not_bool(n) or n < 1:
            raise ValueError(f"_release_reservations_locked requires n >= 1 (int), got {n!r}")
        if self._size < n:
            logger.error(
                "pool: _release_reservations_locked called with _size=%d, n=%d; "
                "ignoring to keep accounting non-negative. This indicates a "
                "double-release bug — check recent changes to the cancel/cleanup paths.",
                self._size,
                n,
            )
            # Wake any parked acquirer: the double-release may have lost a prior
            # signal, so re-evaluate against the current _size.
            self._signal_state_change()
            return False
        self._size -= n
        return True

    async def _release_reservation(self) -> None:
        """Decrement ``_size`` by 1 under the lock, waking waiters.

        Every per-conn ``_size -= 1`` goes through here; bulk decrements call
        ``_release_reservations_locked`` directly. Both share the under-flow guard.
        """
        async with self._lock:
            decremented = self._release_reservations_locked(1)
        if decremented:
            self._signal_state_change()

    async def _release_after_drain(self, drain_task: asyncio.Task[None]) -> None:
        """Await an abandoned drain task, then release its slot.

        Holding the slot until the orphan completes keeps acquire() from dialing a
        replacement while the orphan transport is still open (max_size invariant).
        ``suppress(BaseException)`` ensures the slot is always eventually released;
        the close() exception is observed by the call-site done-callback.
        """
        with contextlib.suppress(BaseException):
            await drain_task
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(self._release_reservation())

    def _get_closed_event(self) -> asyncio.Event:
        """Lazily create the closed Event bound to the running loop."""
        if self._closed_event is None:
            self._closed_event = asyncio.Event()
        return self._closed_event

    def _signal_state_change(self) -> None:
        """Wake any acquire() waiters to re-check pool state.

        Reuses the closed-event; acquire() clears it before parking each
        iteration and re-checks _closed at the loop top.
        """
        if self._closed_event is not None:
            self._closed_event.set()

    async def _drain_idle(self, *, deadline: float | None = None) -> None:
        """Close all idle connections in the pool.

        Called when a conn is found broken (leader change, restart) since the rest
        are likely stale too. ``deadline`` (monotonic loop.time()) stops the drain
        between iterations so the caller's pool.timeout is honoured end-to-end;
        ``None`` leaves the per-iteration cap as the only bound.
        """
        deadline_exit = False
        try:
            while not self._pool.empty():
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    logger.debug(
                        "pool: _drain_idle hit the overall deadline; "
                        "leaving remaining idle connections for the "
                        "next acquire / close"
                    )
                    deadline_exit = True
                    return
                try:
                    conn = self._pool.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover
                    # empty() is racy with a concurrent get_nowait; too tight to
                    # drive without monkey-patching asyncio.Queue.
                    break
                # Whether the slot release was deferred to _release_after_drain
                # (set by the TimeoutError arm); read by the finally.
                drain_deferred = False
                try:
                    # Clear _pool_released BEFORE close or close() early-returns
                    # (a checked-in conn no-ops) and the writer leaks.
                    conn._pool_released = False
                    # Shield each close against an outer asyncio.timeout(close()):
                    # a cancel mid-wait_closed would otherwise orphan every
                    # subsequent queued conn. Explicit Task + observer so the
                    # implicit shield task isn't left unobserved on TimeoutError.
                    # See _drain_remaining_after_cancel for the cap rationale.
                    inner_drain = asyncio.ensure_future(conn.close())
                    inner_drain.add_done_callback(_observe_drain_exception)
                    # Clip the per-iter cap to the remaining deadline: the top-of-
                    # loop gate only fires between iterations, so a stuck close
                    # could otherwise overshoot by (multiplier-1) × close_timeout.
                    per_iter_cap = self._close_timeout * _DRAIN_PER_CONN_CAP_MULTIPLIER
                    if deadline is not None:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            logger.debug(
                                "pool: _drain_idle deadline exhausted mid-iteration; abandoning"
                            )
                            deadline_exit = True
                            return
                        per_iter_cap = min(per_iter_cap, remaining)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(inner_drain),
                            timeout=per_iter_cap,
                        )
                    except TimeoutError:
                        # Inner close is buggy or pathologically stuck. Defer the
                        # slot release until the orphan completes so a concurrent
                        # acquire() can't dial a replacement past max_size; the
                        # done-callback observes the orphan's eventual exception.
                        logger.warning(
                            "pool: close() on idle connection %s exceeded "
                            "per-iteration drain cap; abandoning to continue queue",
                            sanitize_for_log(str(getattr(conn, "_address", "?"))),
                        )
                        drain_deferred = True
                        conn._pool_released = True
                        # Hold a strong ref so GC can't reclaim it mid-flight.
                        _release_task = asyncio.ensure_future(
                            self._release_after_drain(inner_drain)
                        )
                        self._background_tasks.add(_release_task)
                        _release_task.add_done_callback(self._background_tasks.discard)
                except Exception:
                    # Transport failures absorbed so the drain finishes the rest;
                    # cancels / KeyboardInterrupt / SystemExit propagate.
                    logger.debug(
                        "pool: close() on idle connection %s failed",
                        sanitize_for_log(str(getattr(conn, "_address", "?"))),
                        exc_info=True,
                    )
                finally:
                    if not drain_deferred:
                        # Restore the flag (True ⇔ pool owns the close path) so a
                        # stale-reference second close() takes the early-return.
                        conn._pool_released = True
                        # Shield the decrement against an outer cancel during the
                        # lock acquire, else _size drifts above capacity.
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.shield(self._release_reservation())
        finally:
            # A cancel out of one iteration aborts the loop and leaves remaining
            # conns un-closed; sweep the queue under shield so none is orphaned.
            # Skip on deadline_exit — the caller's deadline expired and an
            # unbounded sweep would defeat it; the queue waits for the next caller.
            if not deadline_exit:
                with contextlib.suppress(asyncio.CancelledError):
                    # Forward the deadline so the recovery sweep stays in budget.
                    await asyncio.shield(self._drain_remaining_after_cancel(deadline=deadline))

    async def _drain_remaining_after_cancel(self, *, deadline: float | None = None) -> None:
        """Best-effort sweep for conns still queued after the main drain exited.

        Each conn is closed and its slot released; failures are absorbed and
        re-entry is safe (the queue empties on the first sweep). ``deadline``
        (monotonic loop.time()) stops it between iterations so close() stays in the
        operator's SIGTERM budget.
        """
        loop = asyncio.get_running_loop()
        while not self._pool.empty():
            if deadline is not None and loop.time() >= deadline:
                # Remaining queue stays for the next caller / finalizer.
                logger.debug(
                    "pool: _drain_remaining_after_cancel deadline "
                    "exhausted between iterations; abandoning"
                )
                return
            try:
                conn = self._pool.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover
                break
            # Clear _pool_released BEFORE close or close() early-returns (checked-in
            # conns no-op) and the transport + reader task leak; restore after.
            conn._pool_released = False
            # Set by the TimeoutError arm to defer slot release until the orphan
            # drain finishes; read by the finally.
            drain_deferred = False
            try:
                # Shield + per-iter cap so a stuck close doesn't block the whole
                # loop and a cancel doesn't leak the rest of the queue. Explicit
                # Task + observer so the implicit shield task is observed on
                # timeout. See _drain_idle for full rationale.
                inner_drain = asyncio.ensure_future(conn.close())
                inner_drain.add_done_callback(_observe_drain_exception)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(inner_drain),
                        timeout=self._close_timeout * _DRAIN_PER_CONN_CAP_MULTIPLIER,
                    )
                except TimeoutError:
                    logger.warning(
                        "pool: cleanup-after-cancel close() on %s "
                        "exceeded close_timeout; abandoning to drain "
                        "remaining queue",
                        sanitize_for_log(str(getattr(conn, "_address", "?"))),
                    )
                    # Defer slot release until the orphan finishes (max_size
                    # invariant); see _drain_idle's TimeoutError arm.
                    drain_deferred = True
                    conn._pool_released = True
                    # Hold a strong ref so GC can't reclaim it mid-flight.
                    _release_task = asyncio.ensure_future(self._release_after_drain(inner_drain))
                    self._background_tasks.add(_release_task)
                    _release_task.add_done_callback(self._background_tasks.discard)
            except Exception:
                logger.debug(
                    "pool: cleanup-after-cancel close() on %s failed",
                    sanitize_for_log(str(getattr(conn, "_address", "?"))),
                    exc_info=True,
                )
            finally:
                if not drain_deferred:
                    # Restore the flag so a stale-reference second close() no-ops.
                    conn._pool_released = True
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Acquire a connection from the pool."""
        if _conn_mod.get_current_pid() != self._creator_pid:
            raise InterfaceError(
                f"Pool used after fork; reconstruct from configuration "
                f"in the target process. (created in pid {self._creator_pid}, "
                f"current pid {_conn_mod.get_current_pid()})"
            )
        self._check_loop_binding()
        if self._closed:
            raise DqliteConnectionError(f"Pool is closed (id={id(self)})")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        conn: DqliteConnection | None = None

        while conn is None:
            if self._closed:
                raise DqliteConnectionError(f"Pool is closed (id={id(self)})")

            try:
                conn = self._pool.get_nowait()
                break
            except asyncio.QueueEmpty:
                pass

            # Reserve a slot under the lock, then drop the lock before the
            # handshake so concurrent users aren't serialized on network latency.
            # try: opens BEFORE the lock so the increment shares the frame whose
            # except arm releases it (a BaseException between lock-exit and try:
            # would otherwise leak the slot).
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
                    # Clamp the create await to the remaining acquire deadline:
                    # ClusterClient.connect's retry budget can run for tens of
                    # seconds past the user's pool.timeout. Same clamp below.
                    create_remaining = deadline - loop.time()
                    if create_remaining <= 0:
                        # Deadline already past: surface the public error with an
                        # actionable cause regardless of which phase exhausted it.
                        idle = self._pool.qsize()
                        checked_out = self._size - idle
                        raise DqliteConnectionError(
                            f"Timed out creating a fresh connection from the pool "
                            f"(pool_id={id(self)}, max_size={self._max_size}, "
                            f"checked_out={checked_out}, idle={idle}, "
                            f"timeout={self._timeout}s)."
                        ) from TimeoutError(
                            f"acquire deadline already exceeded by "
                            f"{loop.time() - deadline:.3f}s before "
                            f"_create_connection ran"
                        )
                    try:
                        async with asyncio.timeout(create_remaining):
                            conn = await self._create_connection()
                    except TimeoutError as exc:
                        # Translate the clamp's bare TimeoutError here (not in the
                        # broader except below) so the cause chain carries
                        # actionable text and names the fresh-dial phase.
                        idle = self._pool.qsize()
                        checked_out = self._size - idle
                        raise DqliteConnectionError(
                            f"Timed out creating a fresh connection from the pool "
                            f"(pool_id={id(self)}, max_size={self._max_size}, "
                            f"checked_out={checked_out}, idle={idle}, "
                            f"timeout={self._timeout}s)."
                        ) from exc
            except BaseException:
                # Shield the release so a cancel re-arming on the await checkpoint
                # can't bypass the decrement (each leak wedges the pool at
                # max_size). if reserved: avoids double-release on the pre-grant
                # _closed raise.
                if reserved:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
                raise
            if reserved:
                # conn is assigned on the success path; except re-raises.
                assert conn is not None
                # close() may have run while _create_connection was suspended;
                # without this re-check the conn would be yielded on a closed pool
                # (contract violation + silent leak). Shield close and release so a
                # cancel can't leak the transport or slot. Symmetric below.
                if self._closed:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(conn.close())
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
                    raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
                break

            # At capacity — wait briefly on the queue, then re-check capacity.
            remaining = deadline - loop.time()
            if remaining <= 0:
                # _size is checked-out + reserved; qsize is idle.
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
            # Race the queue against the state-change event so any pool change
            # wakes waiters promptly. The check-then-clear runs under the lock so
            # a concurrent close() can't set() the event between read and clear.
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
                # Create both tasks inside the try so a cancel between the two
                # create_task calls can't orphan one — the finally cancels both.
                get_task = asyncio.create_task(self._pool.get())
                closed_task = asyncio.create_task(closed_event.wait())
                done, _pending = await asyncio.wait(
                    {get_task, closed_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                # asyncio.wait doesn't cancel its argument tasks, so on outer
                # cancel both children are still alive — stop them before
                # propagating, else an abandoned get_task can win a later put()
                # and orphan a connection.
                if closed_task is not None and not closed_task.done():
                    closed_task.cancel()
                    # Await the cancelled task so its CancelledError doesn't sit
                    # until GC; narrow so KeyboardInterrupt/SystemExit propagate.
                    with contextlib.suppress(asyncio.CancelledError):
                        await closed_task
                if (
                    get_task is not None
                    and get_task.done()
                    and not get_task.cancelled()
                    and get_task.exception() is None
                ):
                    # Cancel raced a successful get; return the conn to the queue
                    # so the next acquirer reuses it rather than shrinking _size.
                    # Absorb RuntimeError ("Event loop is closed" under racing
                    # engine.dispose()) so the bare raise below re-raises the
                    # original cancel; narrow so programming bugs still surface.
                    try:
                        await self._put_back_or_release_late_winner(get_task.result())
                    except RuntimeError:
                        logger.debug(
                            "pool.acquire cleanup: late-winner helper raised "
                            "RuntimeError (typically 'Event loop is closed' "
                            "under engine.dispose()); original cancel/"
                            "exception preserved via __context__.",
                            exc_info=True,
                        )
                elif get_task is not None and not get_task.done():
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                elif get_task is not None and get_task.done():
                    # Completed with cancellation/exception: consume it so it
                    # doesn't log at GC; suppress BaseException since the consumed
                    # exception may be a CancelledError and the outer cancel wins.
                    with contextlib.suppress(BaseException):
                        await get_task
                raise
            assert get_task is not None and closed_task is not None
            if not closed_task.done():
                closed_task.cancel()
                # Await so the CancelledError doesn't sit until GC; narrow so
                # KeyboardInterrupt/SystemExit propagate.
                with contextlib.suppress(asyncio.CancelledError):
                    await closed_task
            if get_task.done() and not get_task.cancelled() and get_task.exception() is None:
                # Live-state re-check: a sibling _release can put_nowait and
                # resolve get_task during the await above. Trusting the earlier
                # done-snapshot would route the winning conn into the discard arm
                # and leak a slot.
                conn = get_task.result()
            else:
                # close or the poll timer fired; cancel the wait and re-check.
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                if get_task.done() and not get_task.cancelled() and get_task.exception() is None:
                    # Cancel raced a successful get; route the conn back so the
                    # slot isn't leaked. Absorb RuntimeError (loop closed under
                    # engine.dispose()) so the loop's continue isn't supplanted.
                    try:
                        await self._put_back_or_release_late_winner(get_task.result())
                    except RuntimeError:
                        logger.debug(
                            "pool.acquire post-wait demux: late-winner helper "
                            "raised RuntimeError (typically 'Event loop is "
                            "closed' under engine.dispose())",
                            exc_info=True,
                        )
                continue

        # If the conn is dead, discard, drain the other (likely-stale) idle
        # conns, and dial a fresh one. is_connected is the protocol flag;
        # _socket_looks_dead is a cheap transport peek that catches a conn which
        # saw a clean peer FIN (leader flip) yet still reports is_connected.
        if not conn.is_connected or _socket_looks_dead(conn):
            logger.debug(
                "pool.acquire: drain-idle triggered by stale conn=%r closing_idle=%d",
                conn,
                self._pool.qsize(),
            )
            # Close this dequeued conn explicitly: _drain_idle only walks the idle
            # queue, so a half-closed conn here would leak its writer. Clear
            # _pool_released first or the close no-ops; shielded against cancel.
            conn._pool_released = False
            try:
                try:
                    await self._close_best_effort(conn, "acquire-drain-stale-conn")
                except RuntimeError:
                    # "Event loop is closed" / cross-loop Lock during a loop-
                    # shutdown race. Proceed with cleanup so the slot is released;
                    # without this catch _size permanently inflates by one.
                    logger.debug(
                        "pool.acquire: dead-conn close raised RuntimeError "
                        "(loop-shutdown race); proceeding with cleanup",
                        exc_info=True,
                    )
                except BaseException:
                    # Propagate but release the reservation first so _size stays
                    # consistent; shield so an outer cancel can't strand it.
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
                    raise
            finally:
                # Restore the flag so a stale-reference second close() no-ops.
                conn._pool_released = True
            # Wrap _drain_idle so a cancel mid-drain still releases the dead conn's
            # reservation. Pass the acquire deadline so the drain stays in the
            # pool.timeout budget instead of blaming the wrong phase on timeout.
            try:
                await self._drain_idle(deadline=deadline)
            except BaseException:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())
                raise
            # Flush the leader cache so the next find_leader runs fresh rather than
            # paying a stale-leader RTT per dead-conn discovery (linear in queue
            # depth). getattr tolerates __new__-built test fixtures.
            cluster = getattr(self, "_cluster", None)
            if cluster is not None:
                cluster._set_last_known_leader(None)
            # The dead conn's reservation is reused for the fresh one; no counter
            # change. Clamp the create await to the remaining deadline (as above).
            try:
                create_remaining = deadline - loop.time()
                if create_remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(create_remaining):
                    conn = await self._create_connection()
            except BaseException as exc:
                # Shielded release (as the new-slot arm above).
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
            # close() may have run while _create_connection was suspended; re-check
            # so the conn isn't yielded on a closed pool. Shield close + release.
            if self._closed:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(conn.close())
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())
                raise DqliteConnectionError(f"Pool is closed (id={id(self)})")
            # A freshly-dialed conn reporting is_connected=False means a silent
            # handshake failure / peer tore it between completion and return; raise
            # a clean error rather than yield a zombie. Narrower than the
            # _socket_looks_dead peek above, which false-positives on bare mocks.
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
            # returned_to_queue tracks whether the reservation was transferred to
            # an in-queue conn; if not, the finally releases it. close() no-ops
            # once _pool_released is True, so the pool must close BEFORE setting it.
            returned_to_queue = False
            try:
                if conn.is_connected and not self._closed:
                    # Healthy conn, user code raised a non-connection error: roll
                    # back any open tx, then return to pool.
                    if await self._reset_connection(conn):
                        # Re-check _closed after the ROLLBACK yield: a concurrent
                        # close() (no _lock) may have drained the queue.
                        if self._closed:
                            try:
                                await asyncio.shield(conn.close())
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
                                    await asyncio.shield(conn.close())
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
                            await asyncio.shield(conn.close())
                        except _POOL_CLEANUP_EXCEPTIONS:
                            logger.debug(
                                "pool: ignoring close() error during release",
                                exc_info=True,
                            )
                        conn._pool_released = True
                else:
                    # Broken conn (invalidated by error handlers): drain the other
                    # (likely-stale) idle conns. Bound the drain by close_timeout
                    # since the acquire deadline already fired, else an unbounded
                    # drain delays cancel/exception observation by max_size × cap.
                    cleanup_deadline = asyncio.get_running_loop().time() + self._close_timeout
                    try:
                        await self._drain_idle(deadline=cleanup_deadline)
                    except _POOL_CLEANUP_EXCEPTIONS:
                        logger.debug(
                            "pool.acquire cleanup: _drain_idle failed",
                            exc_info=True,
                        )
                    # Shield conn.close() so an outer cancel mid-cleanup doesn't
                    # leak the transport; _drain_idle above stays bare (shielding
                    # it could turn an outer cancel into an unbounded wait).
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
                            # "Event loop is closed" under engine.dispose(): absorb
                            # so the user's original exception still propagates;
                            # narrow so programming bugs surface.
                            logger.debug(
                                "pool.acquire cleanup: conn.close(%s) raised RuntimeError",
                                sanitize_for_log(str(getattr(conn, "_address", "?"))),
                                exc_info=True,
                            )
                    finally:
                        # Always set the flag so a subsequent close() short-circuits.
                        conn._pool_released = True
            finally:
                if not returned_to_queue:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(self._release_reservation())
            raise
        else:
            # Shield the happy-path release so a cancel at the bare await can't
            # bypass _release's reservation discipline and leave the conn checked
            # out. Under shield _release completes in the background; the slot
            # release becoming async to the cancel observation beats a slot leak.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._release(conn))

    async def _reset_connection(self, conn: DqliteConnection) -> bool:
        """Roll back any open transaction before returning to pool.

        Returns True if the connection is clean and reusable, False if it should
        be destroyed. A raising ROLLBACK leaves tx state unknowable, so the conn
        is dropped; the cluster's Raft log reclaims any uncommitted work.
        """
        # Drop on return if the socket already looks dead, else a wire-poisoned
        # but clean-tx conn churns one wasted slot on the next dequeue.
        if _socket_looks_dead(conn):
            logger.debug(
                "pool: dropping connection %s (socket looks dead on return)",
                sanitize_for_log(str(conn._address)),
            )
            return False
        # Roll back on any tx flag, including _has_untracked_savepoint (a SAVEPOINT
        # whose name the parser couldn't represent but the server still holds).
        # Strict isinstance guards keep MagicMock attrs (truthy) from forcing a
        # ROLLBACK on every fake-conn release.
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
                # "no transaction is active" (e.g. a leader flip where the new
                # leader has no record of our tx) means the conn is healthy:
                # scrub the local flags and preserve the slot.
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
                # Leader-flip (normal churn) logs at DEBUG; a genuine server
                # failure logs at WARNING with traceback.
                code = getattr(exc, "code", None)
                if code in LEADER_ERROR_CODES:
                    logger.debug(
                        "pool: dropping connection %s after leader-class "
                        "ROLLBACK failure (code=%s)",
                        sanitize_for_log(str(conn._address)),
                        code,
                    )
                else:
                    # %s (not %r) on the already-repr-sanitised exc so it isn't
                    # doubly-quoted.
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

    async def _drain_pending_under_shield(self, conn: DqliteConnection) -> None:
        """Snapshot ``conn._pending_drain`` and absorb it under shield.

        A cancel mid-ROLLBACK has _invalidate schedule a bounded wait_closed drain;
        close()'s _pool_released early-return skips awaiting it, so await it here so
        the reader task doesn't outlive the conn. Called at every _release exit
        shape so a _close_impl refactor can't regress drain hygiene at one site.
        Absorb the awaiter-side CancelledError (the shield protects the inner task).
        """
        pending = getattr(conn, "_pending_drain", None)
        if pending is None:
            return
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            # uncancel() the absorbed cancel so outer TaskGroup / asyncio.timeout
            # gates that read Task.cancelling() see an accurate count.
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        except Exception:
            logger.debug(
                "pool _release: suppressed pending-drain exception on conn %s",
                sanitize_for_log(str(getattr(conn, "_address", "?"))),
                exc_info=True,
            )

    async def _release(self, conn: DqliteConnection) -> None:
        """Return a connection to the pool or close it.

        close() no-ops once _pool_released is True, so close MUST run first or the
        transport leaks. The reservation release lives in an outer finally so it
        runs even on a cancel during the ROLLBACK / close awaits.
        """
        # Fork-after-acquire: __aexit__ running in the child. Touching the
        # parent-loop _lock would raise a cross-loop asyncio error; just mark the
        # conn released so its finalizer stays quiet (the child never held the slot).
        if _conn_mod.get_current_pid() != self._creator_pid:
            with contextlib.suppress(AttributeError):
                conn._pool_released = True
            return
        # _check_loop_binding lives INSIDE the try so the finally still releases
        # the slot when the binding check raises on cross-loop misuse — else every
        # misuse-then-retry permanently shrinks max_size by 1.
        returned_to_queue = False
        try:
            self._check_loop_binding()
            if self._closed:
                await self._close_best_effort(conn, "release-closed")
                # Drain _pending_drain BEFORE _pool_released=True so the close-side
                # drain is observed; see _drain_pending_under_shield.
                await self._drain_pending_under_shield(conn)
                conn._pool_released = True
                return

            if not await self._reset_connection(conn):
                await self._close_best_effort(conn, "release-reset-rolled-back")
                await self._drain_pending_under_shield(conn)
                conn._pool_released = True
                return

            # Re-check _closed after the ROLLBACK yield: a concurrent close() (no
            # _lock) may have drained the queue, which would orphan this conn.
            if self._closed:
                await self._close_best_effort(conn, "release-post-reset-closed")
                await self._drain_pending_under_shield(conn)
                conn._pool_released = True
                return

            # Healthy conn: just enqueue (close first if the queue is full).
            # INVARIANT: no await between the _closed re-check above and put_nowait,
            # so a concurrent close() can't drain between them. Do NOT add an await
            # here without a fresh _closed re-check before put_nowait (TOCTOU).
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                await self._close_best_effort(conn, "release-queuefull")
                conn._pool_released = True
            else:
                conn._pool_released = True
                # Reservation transfers to the queued conn; do not decrement.
                returned_to_queue = True
        finally:
            if not returned_to_queue:
                # Reached with _pool_released still False on a cancel during the
                # ROLLBACK / close awaits. Mark it (after draining _pending_drain)
                # so a stale-reference close() takes the early-return.
                await self._drain_pending_under_shield(conn)
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

        Idle conns close immediately; in-use conns close on return via _release's
        _closed branch. close() returns when the IDLE queue is drained, NOT when
        checked-out conns return — there is no wait_drained() API by design (matches
        go-dqlite); cancel/await in-flight tasks first for dispose() semantics.

        Idempotent and concurrent-caller-safe (second callers wait on _close_done).
        The queue drains sequentially, so total wall-clock ~ qsize × close_timeout;
        wrap in asyncio.timeout() for a hard bound. See close_timeout for sizing.
        """
        # Fork-after-init: child shares the parent's conn FDs, so draining here
        # would send FIN on sockets the parent still uses. Flip closed and null the
        # parent-loop primitives so any later child touch trips an is-None guard
        # (not a deep cross-loop asyncio error) and a child initialize()/close()
        # doesn't park on a parent's Event. Runs BEFORE the _closed early-return so
        # a child mid-parent-close doesn't await a parent-loop Event forever.
        if _conn_mod.get_current_pid() != self._creator_pid:
            self._closed = True
            self._closed_flag[0] = True
            self._close_done = None
            self._closed_event = None
            self._initializing = False
            self._initialize_done_event = None
            self._drain_complete = True
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            return
        self._check_loop_binding()
        if self._closed:
            # Second caller waits on the first's drain. (A second caller on a
            # DIFFERENT loop crashes here on the Event loop-binding check — pool
            # lifecycle is single-loop by design.)
            if self._close_done is not None:
                await self._close_done.wait()
            # If the first caller's drain was cut short by an outer cancel,
            # _close_done was set with the queue still under-drained; sweep so this
            # caller honours the "drain completed" contract.
            if not self._drain_complete:
                with contextlib.suppress(asyncio.CancelledError):
                    # Same budget shape as the first caller so this can't extend it.
                    second_close_deadline = asyncio.get_running_loop().time() + (
                        self._close_timeout * self._max_size * _DRAIN_PER_CONN_CAP_MULTIPLIER
                    )
                    await asyncio.shield(
                        self._drain_remaining_after_cancel(deadline=second_close_deadline)
                    )
            return
        # Publish _close_done BEFORE flipping _closed so a second caller observing
        # _closed=True always has a valid Event to wait on (matters only under
        # signal-handler delivery between the two assignments).
        self._close_done = asyncio.Event()
        self._closed = True
        self._closed_flag[0] = True
        try:
            # Detach + drain inside the try so the finally's _close_done.set() is
            # reachable even if a BaseException lands before the first await, else a
            # second caller awaits _close_done forever.
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            logger.debug(
                "pool.close: draining idle=%d in_flight=%d",
                self._pool.qsize(),
                max(self._size - self._pool.qsize(), 0),
            )
            # Aggregate close-budget deadline so a queue of stuck-close peers
            # can't hold the close path for the full N × per_iter_cap. The
            # _drain_idle cancel-recovery arm forwards the same deadline.
            close_deadline = asyncio.get_running_loop().time() + (
                self._close_timeout * self._max_size * _DRAIN_PER_CONN_CAP_MULTIPLIER
            )
            await self._drain_idle(deadline=close_deadline)
            # Set BEFORE the finally so a cancel between the drain return and here
            # leaves _drain_complete=False and siblings run the best-effort sweep.
            self._drain_complete = True
        finally:
            # Both event-sets in finally so a BaseException after _closed=True
            # can't leave acquirers parked until their acquire timeout fires.
            if self._closed_event is not None:
                self._closed_event.set()
            self._close_done.set()
            # _closed_event has no remaining waiters once set (new acquirers
            # short-circuit), so drop it; _close_done stays for second callers.
            self._closed_event = None

        # In-use conns close via acquire()'s cleanup on return; force-closing them
        # here would race the acquire context manager and corrupt _size.

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
