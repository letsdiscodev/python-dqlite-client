"""Connection pooling for dqlite."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


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
    try:
        closing = transport.is_closing() if transport is not None else False
    except Exception:
        closing = False
    try:
        eof = reader.at_eof() if reader is not None else False
    except Exception:
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
        addresses: list[str],
        *,
        database: str = "default",
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 10.0,
    ) -> None:
        """Initialize connection pool.

        Args:
            addresses: List of node addresses
            database: Database name
            min_size: Minimum connections to maintain
            max_size: Maximum connections allowed
            timeout: Connection timeout
        """
        if min_size < 0:
            raise ValueError(f"min_size must be non-negative, got {min_size}")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) must not exceed max_size ({max_size})")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self._addresses = addresses
        self._database = database
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout

        self._cluster = ClusterClient.from_addresses(addresses, timeout=timeout)
        self._pool: asyncio.Queue[DqliteConnection] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._closed_event: asyncio.Event | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the pool with minimum connections."""
        async with self._lock:
            if self._initialized:
                return
            for _ in range(self._min_size):
                conn = await self._create_connection()
                await self._pool.put(conn)
            self._initialized = True

    async def _create_connection(self) -> DqliteConnection:
        """Create a new connection to the leader."""
        conn = await self._cluster.connect(database=self._database)
        self._size += 1
        return conn

    def _get_closed_event(self) -> asyncio.Event:
        """Lazily create the closed Event bound to the running loop."""
        if self._closed_event is None:
            self._closed_event = asyncio.Event()
            if self._closed:
                self._closed_event.set()
        return self._closed_event

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
            except BaseException:
                pass
            finally:
                self._size -= 1

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

            # Try to create a new connection if under max
            async with self._lock:
                if self._closed:
                    raise DqliteConnectionError("Pool is closed")
                if self._size < self._max_size:
                    conn = await self._create_connection()
                    break

            # At capacity — wait briefly on the queue, then loop back to
            # re-check capacity (another coroutine may have freed a slot)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise DqliteConnectionError(
                    f"Timed out waiting for a connection from the pool "
                    f"(max_size={self._max_size}, timeout={self._timeout}s)"
                )
            # Race the queue against the closed event so close() wakes
            # waiters promptly instead of leaving them on the polling loop.
            closed_event = self._get_closed_event()
            get_task: asyncio.Task[DqliteConnection] = asyncio.create_task(self._pool.get())
            closed_task = asyncio.create_task(closed_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    {get_task, closed_task},
                    timeout=min(remaining, 0.5),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
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
            async with self._lock:
                self._size -= 1
                conn = await self._create_connection()

        conn._pool_released = False
        try:
            yield conn
        except BaseException:
            if conn.is_connected and not self._closed:
                # Connection is healthy — user code raised a non-connection error.
                # Roll back any open transaction, then return to pool.
                if not await self._reset_connection(conn):
                    conn._pool_released = True
                    with contextlib.suppress(Exception):
                        await conn.close()
                    self._size -= 1
                    raise
                conn._pool_released = True
                try:
                    self._pool.put_nowait(conn)
                except asyncio.QueueFull:
                    await conn.close()
                    self._size -= 1
            else:
                # Connection is broken (invalidated by execute/fetch error handlers).
                # Drain other idle connections — they likely point to the same dead server.
                conn._pool_released = True
                await self._drain_idle()
                with contextlib.suppress(BaseException):
                    await conn.close()
                self._size -= 1
            raise
        else:
            await self._release(conn)

    async def _reset_connection(self, conn: DqliteConnection) -> bool:
        """Roll back any open transaction before returning to pool.

        Returns True if the connection is clean and can be reused,
        False if it should be destroyed.
        """
        if conn._in_transaction:
            # Cheap pre-write liveness check: if the transport is already
            # closing or the reader has seen EOF, ROLLBACK would stall on
            # _read_data until self._timeout. Bail fast instead.
            if _socket_looks_dead(conn):
                return False
            try:
                await conn.execute("ROLLBACK")
            except BaseException:
                return False
            conn._in_transaction = False
            conn._tx_owner = None
        return True

    async def _release(self, conn: DqliteConnection) -> None:
        """Return a connection to the pool or close it."""
        if self._closed:
            conn._pool_released = True
            await conn.close()
            self._size -= 1
            return

        if not await self._reset_connection(conn):
            conn._pool_released = True
            with contextlib.suppress(Exception):
                await conn.close()
            self._size -= 1
            return

        conn._pool_released = True
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
            self._size -= 1

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
        # returning to the pool.  Force-closing them here would race with
        # the acquire context manager and corrupt _size (see #080).

    async def __aenter__(self) -> "ConnectionPool":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
