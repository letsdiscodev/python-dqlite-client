"""Connection pooling for dqlite."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import ConnectionError


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
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def initialize(self) -> None:
        """Initialize the pool with minimum connections."""
        for _ in range(self._min_size):
            conn = await self._create_connection()
            await self._pool.put(conn)

    async def _create_connection(self) -> DqliteConnection:
        """Create a new connection to the leader."""
        conn = await self._cluster.connect(database=self._database)
        self._size += 1
        return conn

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DqliteConnection]:
        """Acquire a connection from the pool."""
        if self._closed:
            raise ConnectionError("Pool is closed")

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
            conn = await self._pool.get()

        try:
            # Verify connection is still good
            if not conn.is_connected:
                await conn.connect()

            yield conn
        except Exception:
            # On error, close connection and create new one
            import contextlib

            with contextlib.suppress(Exception):
                await conn.close()
            self._size -= 1
            raise
        else:
            # Return to pool
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                # Pool full, close connection
                await conn.close()
                self._size -= 1

    async def execute(self, sql: str, params: list[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.execute(sql, params)

    async def fetch(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query using a pooled connection."""
        async with self.acquire() as conn:
            return await conn.fetch(sql, params)

    async def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True

        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break

        self._size = 0

    async def __aenter__(self) -> "ConnectionPool":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
