"""Connection pooling for dqlite."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import NodeStore
from dqliteclient.protocol import _validate_positive_int_or_none

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
        max_total_rows: int | None = 10_000_000,
        max_continuation_frames: int | None = 100_000,
        trust_server_heartbeat: bool = False,
    ) -> None:
        """Initialize connection pool.

        Args:
            addresses: List of node addresses. Ignored if ``cluster`` or
                ``node_store`` is provided; required otherwise.
            database: Database name
            min_size: Minimum connections to maintain
            max_size: Maximum connections allowed
            timeout: Connection timeout
            cluster: Externally-owned ClusterClient. Lets multiple pools
                share one ClusterClient (and thus its node store, leader
                cache, etc.) across databases or tenants.

                Ownership: the pool does NOT take ownership of this
                cluster. The caller is responsible for eventually calling
                ``cluster.close()`` and MUST NOT close the cluster while
                any pool is still in use. ``pool.close()`` does not close
                the cluster — it only drains pool-owned connections.

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
        """
        if min_size < 0:
            raise ValueError(f"min_size must be non-negative, got {min_size}")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) must not exceed max_size ({max_size})")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
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
        self._initialized = False

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
                self._size += self._min_size
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
                    # Close the connections that did succeed — they are
                    # unowned now that initialize is aborting.
                    for conn in successes:
                        with contextlib.suppress(BaseException):
                            await conn.close()
                    self._size -= self._min_size
                    self._signal_state_change()
                    # Re-raise the first observed failure as the root
                    # cause; additional failures chain as context.
                    raise failures[0]
                for conn in successes:
                    await self._pool.put(conn)
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
        )

    async def _release_reservation(self) -> None:
        """Decrement ``_size`` under the lock, waking waiters.

        Every ``_size -= 1`` call in the pool must go through this
        helper so the counter stays consistent against concurrent
        capacity checks in ``acquire``.
        """
        async with self._lock:
            self._size -= 1
        self._signal_state_change()

    def _get_closed_event(self) -> asyncio.Event:
        """Lazily create the closed Event bound to the running loop."""
        if self._closed_event is None:
            self._closed_event = asyncio.Event()
            if self._closed:
                self._closed_event.set()
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
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await conn.close()
            except Exception as exc:
                # Transport-level failures (BrokenPipeError, OSError, our
                # own DqliteConnectionError) are absorbed so drain can
                # finish the remaining connections. CancelledError /
                # KeyboardInterrupt / SystemExit propagate — swallowing
                # them used to break structured concurrency (``asyncio.
                # timeout`` around ``pool.close()`` would silently hang).
                logger.debug(
                    "pool: close() on idle connection %r failed: %r",
                    getattr(conn, "_address", "?"),
                    exc,
                )
            finally:
                await self._release_reservation()

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
            get_task: asyncio.Task[DqliteConnection] = asyncio.create_task(self._pool.get())
            closed_task = asyncio.create_task(closed_event.wait())
            try:
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
                if not closed_task.done():
                    closed_task.cancel()
                if get_task.done() and not get_task.cancelled() and get_task.exception() is None:
                    # Outer cancel raced with a successful get. The
                    # reservation that backed this connection is still
                    # valid; return it to the queue so the next
                    # acquirer can use it instead of closing and
                    # releasing (which would shrink _size).
                    conn_won = get_task.result()
                    with contextlib.suppress(asyncio.QueueFull):
                        self._pool.put_nowait(conn_won)
                elif not get_task.done():
                    get_task.cancel()
                    with contextlib.suppress(BaseException):
                        await get_task
                raise
            if not closed_task.done():
                closed_task.cancel()
            if get_task in done:
                conn = get_task.result()
            else:
                # Either close fired or the poll timer fired; either way,
                # cancel the queue wait cleanly and let the loop re-check.
                get_task.cancel()
                with contextlib.suppress(BaseException):
                    await get_task
                continue

        # If connection is dead, discard and create a fresh one with leader discovery.
        # Also drain other idle connections — they likely point to the same dead server.
        if not conn.is_connected:
            await self._drain_idle()
            # Release the dead reservation, reserve a new slot, then
            # do the network work outside the lock.
            async with self._lock:
                self._size -= 1  # dead conn's reservation
                self._size += 1  # fresh reservation
            try:
                conn = await self._create_connection()
            except BaseException:
                await self._release_reservation()
                raise

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
                    with contextlib.suppress(BaseException):
                        await self._drain_idle()
                    with contextlib.suppress(BaseException):
                        await conn.close()
                    conn._pool_released = True
            finally:
                if not returned_to_queue:
                    with contextlib.suppress(BaseException):
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
        if conn._in_transaction:
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
                await conn.execute("ROLLBACK")
            except BaseException as exc:
                logger.debug(
                    "pool: dropping connection %s after ROLLBACK failure: %r",
                    conn._address,
                    exc,
                )
                return False
            conn._in_transaction = False
            conn._tx_owner = None
        return True

    async def _release(self, conn: DqliteConnection) -> None:
        """Return a connection to the pool or close it.

        ``conn.close()`` has an early-return guard against
        ``_pool_released=True``, so close MUST run before the flag is
        set — otherwise the transport leaks (a bug that affects every
        branch that closes a pool-owned connection).
        """
        if self._closed:
            await conn.close()
            conn._pool_released = True
            await self._release_reservation()
            return

        if not await self._reset_connection(conn):
            with contextlib.suppress(Exception):
                await conn.close()
            conn._pool_released = True
            await self._release_reservation()
            return

        # Healthy connection returning to queue: no close; just flip
        # the flag and enqueue. If the queue is full we must close
        # before setting the flag.
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
            conn._pool_released = True
            await self._release_reservation()
        else:
            conn._pool_released = True

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
        """
        self._closed = True
        if self._closed_event is not None:
            self._closed_event.set()
        await self._drain_idle()

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
