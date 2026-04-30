"""Async Python client for dqlite.

Thread safety: connections and pools are NOT thread-safe. All operations
must be performed within a single asyncio event loop. To submit work from
other threads, use ``asyncio.run_coroutine_threadsafe()``. Free-threaded
Python (no-GIL) is not supported — the guard lives in
``dqlitewire.__init__`` (an unconditional transitive dependency via the
``from dqliteclient.connection import DqliteConnection`` chain) and
raises ``ImportError`` at import time.
"""

from dqliteclient.cluster import ClusterClient, RedirectPolicy, allowlist_policy
from dqliteclient.connection import DqliteConnection, parse_address
from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DataError,
    DqliteConnectionError,
    DqliteError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, NodeStore
from dqliteclient.pool import ConnectionPool
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)

__version__ = "0.1.3"

__all__ = [
    "ClusterClient",
    "ClusterError",
    "ClusterPolicyError",
    "ConnectionPool",
    "DataError",
    "DqliteConnection",
    "DqliteConnectionError",
    "DqliteError",
    "InterfaceError",
    "MemoryNodeStore",
    "NodeInfo",
    "NodeStore",
    "OperationalError",
    "ProtocolError",
    "RedirectPolicy",
    "parse_address",
    "__version__",
    "allowlist_policy",
    "connect",
    "create_pool",
]


async def connect(
    address: str,
    *,
    database: str = "default",
    timeout: float = 10.0,
    max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
    max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
    trust_server_heartbeat: bool = False,
    close_timeout: float = 0.5,
) -> DqliteConnection:
    """Connect to a dqlite node.

    Args:
        address: Node address in "host:port" format
        database: Database name to open
        timeout: Per-RPC-phase timeout in seconds (forwarded). Each
            phase of an operation gets the full budget independently;
            wrap with ``asyncio.timeout(...)`` to enforce an end-to-end
            deadline.
        max_total_rows: Cumulative row cap across continuation frames
            for a single query. Forwarded to the underlying
            DqliteConnection. None disables the cap.
        max_continuation_frames: Per-query continuation-frame cap.
            Forwarded to the underlying DqliteConnection.
        trust_server_heartbeat: Let the server-advertised heartbeat
            widen the per-read deadline. Default False.
        close_timeout: Budget (seconds) for the transport-drain during
            ``close()``. After ``writer.close()`` the local side of
            the socket is gone; ``wait_closed`` is best-effort cleanup.
            The 0.5s default is sized for LAN; increase for WAN
            deployments where FIN/ACK round-trip is slower, or
            decrease to tighten SIGTERM-shutdown budgets. See
            ``DqliteConnection.__init__`` for full rationale.

    Returns:
        A connected DqliteConnection
    """
    conn = DqliteConnection(
        address,
        database=database,
        timeout=timeout,
        max_total_rows=max_total_rows,
        max_continuation_frames=max_continuation_frames,
        trust_server_heartbeat=trust_server_heartbeat,
        close_timeout=close_timeout,
    )
    await conn.connect()
    return conn


async def create_pool(
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
) -> ConnectionPool:
    """Create a connection pool with automatic leader detection.

    Args:
        addresses: List of node addresses in "host:port" format. Ignored if
            ``cluster`` or ``node_store`` is provided.
        database: Database name to open
        min_size: Minimum number of connections to maintain
        max_size: Maximum number of connections
        timeout: Per-RPC-phase timeout in seconds (forwarded). Each
            phase of an operation gets the full budget independently;
            wrap with ``asyncio.timeout(...)`` to enforce an end-to-end
            deadline.
        cluster: Externally-owned ClusterClient shared across pools.
        node_store: Externally-owned NodeStore used to build a new
            ClusterClient. Mutually exclusive with ``cluster``.
        max_total_rows: Cumulative row cap across continuation frames
            for a single query. Forwarded to the underlying
            ConnectionPool. None disables the cap.
        max_continuation_frames: Per-query continuation-frame cap.
            Forwarded to the underlying ConnectionPool.
        trust_server_heartbeat: Let the server-advertised heartbeat
            widen the per-read deadline. Default False.
        close_timeout: Budget (seconds) for the transport-drain during
            ``close()``. After ``writer.close()`` the local side of
            the socket is gone; ``wait_closed`` is best-effort cleanup.
            The 0.5s default is sized for LAN; increase for WAN
            deployments where FIN/ACK round-trip is slower, or
            decrease to tighten SIGTERM-shutdown budgets. See
            ``DqliteConnection.__init__`` for full rationale.
        max_attempts: Maximum leader-discovery attempts per pool
            connect (forwarded to ``ClusterClient.connect``). ``None``
            (default) uses the cluster client's default of 3. Must be
            ``>= 1`` if not ``None``.

    Returns:
        An initialized ConnectionPool
    """
    pool = ConnectionPool(
        addresses,
        database=database,
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        cluster=cluster,
        node_store=node_store,
        max_total_rows=max_total_rows,
        max_continuation_frames=max_continuation_frames,
        trust_server_heartbeat=trust_server_heartbeat,
        close_timeout=close_timeout,
        max_attempts=max_attempts,
    )
    await pool.initialize()
    return pool
