"""Async Python client for dqlite.

Thread safety: connections and pools are NOT thread-safe. All operations
must be performed within a single asyncio event loop. To submit work from
other threads, use ``asyncio.run_coroutine_threadsafe()``. Free-threaded
Python (no-GIL) is not supported.
"""

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import (
    ClusterError,
    DataError,
    DqliteConnectionError,
    DqliteError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, NodeStore
from dqliteclient.pool import ConnectionPool

__all__ = [
    "connect",
    "create_pool",
    "DqliteConnection",
    "ConnectionPool",
    "ClusterClient",
    "NodeStore",
    "NodeInfo",
    "MemoryNodeStore",
    "DqliteError",
    "DqliteConnectionError",
    "InterfaceError",
    "ProtocolError",
    "ClusterError",
    "OperationalError",
    "DataError",
]

__version__ = "0.1.0"


async def connect(
    address: str,
    *,
    database: str = "default",
    timeout: float = 10.0,
) -> DqliteConnection:
    """Connect to a dqlite node.

    Args:
        address: Node address in "host:port" format
        database: Database name to open
        timeout: Connection timeout in seconds

    Returns:
        A connected DqliteConnection
    """
    conn = DqliteConnection(address, database=database, timeout=timeout)
    await conn.connect()
    return conn


async def create_pool(
    addresses: list[str],
    *,
    database: str = "default",
    min_size: int = 1,
    max_size: int = 10,
    timeout: float = 10.0,
) -> ConnectionPool:
    """Create a connection pool with automatic leader detection.

    Args:
        addresses: List of node addresses in "host:port" format
        database: Database name to open
        min_size: Minimum number of connections to maintain
        max_size: Maximum number of connections
        timeout: Connection timeout in seconds

    Returns:
        An initialized ConnectionPool
    """
    pool = ConnectionPool(
        addresses,
        database=database,
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
    )
    await pool.initialize()
    return pool
