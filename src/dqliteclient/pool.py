"""Connection pooling for dqlite."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError


class ConnectionPool:
    """Connection pool with automatic leader detection."""

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
            raise ValueError(
                f"min_size ({min_size}) must not exceed max_size ({max_size})"
            )
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self._addresses = addresses
        self._database = database
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout

        self._cluster = ClusterClient.from_addresses(addresses, timeout=timeout)
        self._pool: asyncio.Queue[DqliteConnection] = asyncio.Queue(maxsize=max_size)
        self._in_use: set[DqliteConnection] = set()
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False
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

    async def _drain_idle(self) -> None:
        """Close all idle connections in the pool.

        Called when a connection is found to be broken (e.g., after a
        leader change or server restart), since other idle connections
        are likely stale too.
        """
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                with contextlib.suppress(Exception):
                    await conn.close()
                self._size -= 1
            except asyncio.QueueEmpty:
                break

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise DqliteConnectionError("Pool is closed")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        conn: DqliteConnection | None = None

        while conn is None:
            # Try to get an idle connection from the queue
            try:
                conn = self._pool.get_nowait()
                break
            except asyncio.QueueEmpty:
                pass

            # Try to create a new connection if under max
            async with self._lock:
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
            try:
                conn = await asyncio.wait_for(
                    self._pool.get(), timeout=min(remaining, 0.5)
                )
            except TimeoutError:
                # Don't fail yet — loop back to re-check _size
                continue

        # If connection is dead, discard and create a fresh one with leader discovery.
        # Also drain other idle connections — they likely point to the same dead server.
        if not conn.is_connected:
            await self._drain_idle()
            async with self._lock:
                self._size -= 1
                conn = await self._create_connection()

        self._in_use.add(conn)
        try:
            yield conn
        except BaseException:
            self._in_use.discard(conn)
            if conn.is_connected and not self._closed:
                # Connection is healthy — user code raised a non-connection error.
                # Return it to the pool instead of destroying it.
                try:
                    self._pool.put_nowait(conn)
                except asyncio.QueueFull:
                    await conn.close()
                    self._size -= 1
            else:
                # Connection is broken (invalidated by execute/fetch error handlers).
                # Drain other idle connections — they likely point to the same dead server.
                await self._drain_idle()
                with contextlib.suppress(BaseException):
                    await conn.close()
                self._size -= 1
            raise
        else:
            self._in_use.discard(conn)
            await self._release(conn)

    async def _release(self, conn: DqliteConnection) -> None:
        """Return a connection to the pool or close it."""
        if self._closed:
            await conn.close()
            self._size -= 1
        else:
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

    async def close(self) -> None:
        """Close the pool and all idle connections.

        Sets the pool as closed so no new connections can be acquired.
        Idle connections are closed immediately. In-use connections are
        closed when they are returned (when the acquire() context manager
        exits). To ensure all connections are closed, cancel or await
        in-flight tasks before calling close().
        """
        self._closed = True

        # Close idle connections
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                with contextlib.suppress(Exception):
                    await conn.close()
                self._size -= 1
            except asyncio.QueueEmpty:
                break

        # In-use connections are closed by acquire()'s cleanup when they
        # return — the else branch checks _closed and closes instead of
        # returning to the pool.  Force-closing them here would race with
        # the acquire context manager and corrupt _size (see #080).

    async def __aenter__(self) -> "ConnectionPool":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
