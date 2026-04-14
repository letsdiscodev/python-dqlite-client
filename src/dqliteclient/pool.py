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

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise DqliteConnectionError("Pool is closed")

        conn: DqliteConnection | None = None

        # Try to get from pool
        try:
            conn = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new if under max
            async with self._lock:
                if self._size < self._max_size:
                    conn = await self._create_connection()

        # Wait for one if at max
        if conn is None:
            try:
                conn = await asyncio.wait_for(
                    self._pool.get(), timeout=self._timeout
                )
            except TimeoutError:
                raise DqliteConnectionError(
                    f"Timed out waiting for a connection from the pool "
                    f"(max_size={self._max_size}, timeout={self._timeout}s)"
                ) from None

        # If connection is dead, discard and create a fresh one with leader discovery
        if not conn.is_connected:
            self._size -= 1
            conn = await self._create_connection()

        self._in_use.add(conn)
        try:
            yield conn
        except BaseException:
            # On error (including cancellation), close connection
            with contextlib.suppress(BaseException):
                await conn.close()
            self._in_use.discard(conn)
            self._size -= 1
            raise
        else:
            self._in_use.discard(conn)
            # Return to pool
            if self._closed:
                await conn.close()
                self._size -= 1
            else:
                try:
                    self._pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool full, close connection
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
        """Close all connections (both idle and in-use)."""
        self._closed = True

        # Close idle connections
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break

        # Close in-use connections
        for conn in list(self._in_use):
            with contextlib.suppress(Exception):
                await conn.close()
        self._in_use.clear()

        self._size = 0

    async def __aenter__(self) -> "ConnectionPool":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
