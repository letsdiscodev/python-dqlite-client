"""Connection pooling for dqlite."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import (
    _TRANSACTION_ROLLBACK_SQL,
    DqliteConnection,
    _is_no_tx_rollback_error,
    _validate_timeout,
)
from dqliteclient.exceptions import (
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import NodeStore
from dqliteclient.protocol import _validate_positive_int_or_none
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)
from dqlitewire import LEADER_ERROR_CODES as _LEADER_ERROR_CODES

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
_POOL_CLEANUP_EXCEPTIONS = (
    OSError,
    DqliteConnectionError,
    ProtocolError,
    OperationalError,
    InterfaceError,
)

logger = logging.getLogger(__name__)


def _socket_looks_dead(conn: DqliteConnection) -> bool:
    """Best-effort local detection of a half-closed TCP socket.

    Returns True only on an affirmative bool signal from the transport or
    reader. Mocked / missing attributes default to False (assume alive) so
    the check never produces a false positive against well-behaved peers.
    """
    protocol = conn._protocol
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
        addresses: list[str] | None = None,
        *,
        database: str = "default",
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 10.0,
        cluster: ClusterClient | None = None,
        node_store: NodeStore | None = None,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = 0.5,
        max_attempts: int | None = None,
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
                deadline.
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
        """
        if min_size < 0:
            raise ValueError(f"min_size must be non-negative, got {min_size}")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) must not exceed max_size ({max_size})")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1 if provided, got {max_attempts}")
        _validate_timeout(timeout)
        _validate_timeout(close_timeout, name="close_timeout")
        if cluster is not None and node_store is not None:
            raise ValueError("pass only one of cluster= or node_store=")
        if cluster is None and node_store is None and not addresses:
            raise ValueError("pass one of addresses, cluster, or node_store")

        self._addresses = addresses or []
        self._database = database
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._max_total_rows = _validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = _validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._trust_server_heartbeat = trust_server_heartbeat
        self._close_timeout = close_timeout
        self._max_attempts = max_attempts

        if cluster is not None:
            self._cluster = cluster
        elif node_store is not None:
            self._cluster = ClusterClient(node_store, timeout=timeout)
        else:
            self._cluster = ClusterClient.from_addresses(self._addresses, timeout=timeout)
        self._pool: asyncio.Queue[DqliteConnection] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._closed_event: asyncio.Event | None = None
        self._close_done: asyncio.Event | None = None
        self._initialized = False

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
        """
        # Hold the lock across the gather so a second concurrent
        # initialize() call observes _initialized=True after the first
        # completes and returns without re-creating.
        async with self._lock:
            if self._initialized:
                return
            if self._min_size > 0:
                logger.debug("pool.initialize: requesting %d connections", self._min_size)
                self._size += self._min_size
                # Count reservations that still need to be released. Each
                # successful put into the pool queue "commits" one slot
                # (the connection stays and must remain counted in
                # _size), so unqueued shrinks per iteration. On any
                # abort the finally below releases exactly ``unqueued``
                # slots — the ones that never made it to the queue —
                # and closes the unqueued survivors.
                unqueued = self._min_size
                unqueued_survivors: list[DqliteConnection] = []
                try:
                    # Create min_size connections concurrently so startup
                    # latency doesn't scale with min_size × per-connect RTT.
                    results = await asyncio.gather(
                        *(self._create_connection() for _ in range(self._min_size)),
                        return_exceptions=True,
                    )
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
                        logger.debug(
                            "pool.initialize: aborting after %d/%d creates succeeded; "
                            "closing %d survivors (first failure: %r)",
                            len(successes),
                            self._min_size,
                            len(successes),
                            failures[0],
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
                                "pool.initialize: create_connection %d/%d failed: %r",
                                i + 1,
                                len(failures),
                                exc,
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
                        raise BaseExceptionGroup(
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
                    if unqueued > 0:
                        # Any exit path with uncommitted reservations —
                        # failed gather, raise from _pool.put, outer
                        # CancelledError mid put-loop — must return the
                        # unqueued slots to _size so a subsequent
                        # initialize()/acquire() is not blocked against
                        # a stale counter climbing toward _max_size.
                        self._size -= unqueued
                        self._signal_state_change()
                    # Close any connection that made it past the gather
                    # but never into the queue. Under a clean success
                    # this list is empty; on partial put-loop cancel it
                    # holds the unqueued tail.
                    for conn in unqueued_survivors:
                        try:
                            await conn.close()
                        except _POOL_CLEANUP_EXCEPTIONS:
                            logger.debug(
                                "pool.initialize: unqueued-survivor close error",
                                exc_info=True,
                            )
            # Do not mark initialized if close() landed during the
            # put-loop and we broke out early — otherwise a subsequent
            # initialize() call on a (re-opened) pool short-circuits.
            if not self._closed:
                self._initialized = True

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
        )

    async def _release_reservation(self) -> None:
        """Decrement ``_size`` under the lock, waking waiters.

        Every ``_size -= 1`` call in the pool must go through this
        helper so the counter stays consistent against concurrent
        capacity checks in ``acquire``.

        Defensive underflow guard: every reservation slot corresponds
        to a prior ``self._size += 1`` and the only decrement path
        is this helper, so ``_size <= 0`` here is unreachable under
        correct accounting. The guard is intentionally cheap — a
        future refactor that double-decrements (e.g., a missed
        condition lands two ``_release_reservation()`` calls per
        slot under the cancel-shielding paths) would otherwise
        silently produce a negative ``_size`` that passes every
        ``self._size < self._max_size`` capacity check and expands
        the pool well beyond its bound. Log at ERROR (operators
        should see this immediately) and refuse the decrement to
        keep accounting non-negative; skip the state-change signal
        because the refusal isn't a transition waiters need to
        react to.
        """
        async with self._lock:
            if self._size <= 0:
                logger.error(
                    "pool: _release_reservation called with _size=%d; "
                    "ignoring to keep accounting non-negative. This "
                    "indicates a double-release bug — check recent "
                    "changes to the cancel/cleanup paths.",
                    self._size,
                )
                return
            self._size -= 1
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
                        "pool: close() on idle connection %r failed",
                        getattr(conn, "_address", "?"),
                        exc_info=True,
                    )
                finally:
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
                    "pool: cleanup-after-cancel close() on %r failed",
                    getattr(conn, "_address", "?"),
                    exc_info=True,
                )
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._release_reservation())

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise DqliteConnectionError("Pool is closed")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        conn: DqliteConnection | None = None

        while conn is None:
            if self._closed:
                raise DqliteConnectionError("Pool is closed")

            # Try to get an idle connection from the queue
            try:
                conn = self._pool.get_nowait()
                break
            except asyncio.QueueEmpty:
                pass

            # Try to reserve a new-connection slot under the lock, then
            # drop the lock before the TCP handshake so concurrent
            # pool users aren't serialized on network latency.
            reserved = False
            async with self._lock:
                if self._closed:
                    raise DqliteConnectionError("Pool is closed")
                if self._size < self._max_size:
                    self._size += 1
                    reserved = True
            if reserved:
                try:
                    conn = await self._create_connection()
                except BaseException:
                    await self._release_reservation()
                    raise
                break

            # At capacity — wait briefly on the queue, then loop back to
            # re-check capacity (another coroutine may have freed a slot)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise DqliteConnectionError(
                    f"Timed out waiting for a connection from the pool "
                    f"(max_size={self._max_size}, timeout={self._timeout}s)"
                )
            # Race the queue against the state-change event so any pool
            # state change (close, size decrement, drain) wakes waiters
            # promptly. The check-_closed-then-clear pair runs under
            # the lock so a concurrent close() can't set() the event
            # between our read and our clear.
            async with self._lock:
                closed_event = self._get_closed_event()
                if self._closed:
                    raise DqliteConnectionError("Pool is closed")
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
                    conn_won = get_task.result()
                    try:
                        self._pool.put_nowait(conn_won)
                    except asyncio.QueueFull:
                        # Invariant violation: reservations should track
                        # queue capacity exactly, so a full queue on
                        # return is "impossible." If it happens anyway,
                        # silently dropping the reference would leak a
                        # live reader task and a socket. Close
                        # explicitly and adjust the reservation count
                        # so the pool shrinks cleanly instead of
                        # leaking. Suppression of close's own errors is
                        # narrow — OSError on an already-dead writer is
                        # expected; anything else propagates.
                        with contextlib.suppress(OSError):
                            await conn_won.close()
                        # Route through the helper so the counter
                        # stays lock-protected and sibling acquirers
                        # parked on ``closed_event.wait()`` get
                        # woken via ``_signal_state_change``. Shield
                        # so a nested cancel cannot leave ``_size``
                        # inconsistent.
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.shield(self._release_reservation())
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
            if get_task in done:
                conn = get_task.result()
            else:
                # Either close fired or the poll timer fired; either way,
                # cancel the queue wait cleanly and let the loop re-check.
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                continue

        # If connection is dead, discard and create a fresh one with leader discovery.
        # Also drain other idle connections — they likely point to the same dead server.
        if not conn.is_connected:
            logger.debug(
                "pool.acquire: drain-idle triggered by stale conn=%r closing_idle=%d",
                conn,
                self._pool.qsize(),
            )
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
            # because the intent was unclear.)
            try:
                conn = await self._create_connection()
            except BaseException:
                await self._release_reservation()
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
                raise DqliteConnectionError("Pool is closed")

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
                            with contextlib.suppress(Exception):
                                await conn.close()
                            conn._pool_released = True
                        else:
                            try:
                                self._pool.put_nowait(conn)
                            except asyncio.QueueFull:
                                with contextlib.suppress(Exception):
                                    await conn.close()
                                conn._pool_released = True
                            else:
                                conn._pool_released = True
                                returned_to_queue = True
                    else:
                        with contextlib.suppress(Exception):
                            await conn.close()
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
                                "pool.acquire cleanup: conn.close(%r) failed",
                                getattr(conn, "_address", "?"),
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
                            # still surface — see done/ISSUE-198.
                            logger.debug(
                                "pool.acquire cleanup: conn.close(%r) raised RuntimeError",
                                getattr(conn, "_address", "?"),
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
            # Cheap pre-write liveness check: if the transport is already
            # closing or the reader has seen EOF, ROLLBACK would stall on
            # _read_data until self._timeout. Bail fast instead.
            if _socket_looks_dead(conn):
                logger.debug(
                    "pool: dropping connection %s (socket looks dead before ROLLBACK)",
                    conn._address,
                )
                return False
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
                        conn._address,
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
                if code in _LEADER_ERROR_CODES:
                    logger.debug(
                        "pool: dropping connection %s after leader-class "
                        "ROLLBACK failure (code=%s)",
                        conn._address,
                        code,
                    )
                else:
                    logger.warning(
                        "pool: dropping connection %s after ROLLBACK failure: %r",
                        conn._address,
                        exc,
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
                    with contextlib.suppress(BaseException):
                        await asyncio.shield(pending)
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
        if self._closed:
            if self._close_done is not None:
                await self._close_done.wait()
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
        try:
            logger.debug(
                "pool.close: draining idle=%d in_flight=%d",
                self._pool.qsize(),
                max(self._size - self._pool.qsize(), 0),
            )
            if self._closed_event is not None:
                self._closed_event.set()
            await self._drain_idle()
        finally:
            self._close_done.set()

        # In-use connections are closed by acquire()'s cleanup when they
        # return — the else branch checks _closed and closes instead of
        # returning to the pool. Force-closing them here would race with
        # the acquire context manager and corrupt _size.

    async def __aenter__(self) -> "ConnectionPool":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
