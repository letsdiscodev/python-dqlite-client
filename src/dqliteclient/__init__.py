"""Async Python client for dqlite.

Connections and pools belong to one event loop and are not thread-safe; submit work
from other threads with ``asyncio.run_coroutine_threadsafe()``. Free-threaded Python is
unsupported (guarded in ``dqlitewire``).
"""

import logging
from collections.abc import Sequence as _Sequence
from typing import Final as _Final

from dqliteclient._dial import DialFunc
from dqliteclient._validate import CLOSE_TIMEOUT_FLOOR as _CLOSE_TIMEOUT_FLOOR
from dqliteclient._validate import CLOSE_TIMEOUT_FLOOR_RATIONALE as _CLOSE_TIMEOUT_FLOOR_RATIONALE
from dqliteclient._validate import DEFAULT_CLOSE_TIMEOUT_SECONDS as _DEFAULT_CLOSE_TIMEOUT_SECONDS
from dqliteclient._validate import DEFAULT_TIMEOUT_SECONDS as _DEFAULT_TIMEOUT_SECONDS
from dqliteclient._validate import parse_address, validate_timeout
from dqliteclient.cluster import (
    ClusterClient,
    LeaderInfo,
    NodeMetadata,
    RedirectPolicy,
    allowlist_policy,
    default_safe_redirect_policy,
)
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import (
    AmbiguousCommitError,
    ClusterError,
    ClusterPolicyError,
    DataError,
    DqliteConnectionError,
    DqliteError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, NodeStore, YamlNodeStore
from dqliteclient.pool import ConnectionPool
from dqliteclient.retry import retry_with_backoff
from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES
from dqlitewire import DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS

CLOSE_TIMEOUT_FLOOR: _Final[float] = _CLOSE_TIMEOUT_FLOOR
CLOSE_TIMEOUT_FLOOR_RATIONALE: _Final[str] = _CLOSE_TIMEOUT_FLOOR_RATIONALE
DEFAULT_CLOSE_TIMEOUT_SECONDS: _Final[float] = _DEFAULT_CLOSE_TIMEOUT_SECONDS
DEFAULT_TIMEOUT_SECONDS: _Final[float] = _DEFAULT_TIMEOUT_SECONDS

__version__: _Final[str] = "0.8.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CLOSE_TIMEOUT_FLOOR",
    "CLOSE_TIMEOUT_FLOOR_RATIONALE",
    "DEFAULT_CLOSE_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "AmbiguousCommitError",
    "ClusterClient",
    "ClusterError",
    "ClusterPolicyError",
    "ConnectionPool",
    "DataError",
    "DialFunc",
    "DqliteConnection",
    "DqliteConnectionError",
    "DqliteError",
    "InterfaceError",
    "LeaderInfo",
    "MemoryNodeStore",
    "NodeInfo",
    "NodeMetadata",
    "NodeStore",
    "OperationalError",
    "ProtocolError",
    "RedirectPolicy",
    "YamlNodeStore",
    "__version__",
    "allowlist_policy",
    "connect",
    "create_pool",
    "default_safe_redirect_policy",
    "parse_address",
    "retry_with_backoff",
    "validate_timeout",
]


async def connect(
    address: str,
    *,
    database: str = "default",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    dial_timeout: float | None = None,
    attempt_timeout: float | None = None,
    max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
    max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
    trust_server_heartbeat: bool = False,
    close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
    dial_func: DialFunc | None = None,
    max_message_size: int | None = None,
) -> DqliteConnection:
    """Open a connection to one node. ``timeout`` bounds each RPC phase, not a whole call."""
    conn = DqliteConnection(
        address,
        database=database,
        timeout=timeout,
        dial_timeout=dial_timeout,
        attempt_timeout=attempt_timeout,
        max_total_rows=max_total_rows,
        max_continuation_frames=max_continuation_frames,
        trust_server_heartbeat=trust_server_heartbeat,
        close_timeout=close_timeout,
        dial_func=dial_func,
        max_message_size=max_message_size,
    )
    try:
        await conn.connect()
    except BaseException:
        conn.terminate()
        raise
    return conn


async def create_pool(
    addresses: _Sequence[str] | None = None,
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
    max_attempts: int | None = None,
    max_elapsed_seconds: float | None = None,
    dial_func: DialFunc | None = None,
    concurrent_leader_conns: int | None = None,
    redirect_policy: RedirectPolicy | None = None,
    max_message_size: int | None = None,
) -> ConnectionPool:
    """Create a pool and open its ``min_size`` warm-up connections to the leader."""
    pool = ConnectionPool(
        addresses,
        database=database,
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        dial_timeout=dial_timeout,
        attempt_timeout=attempt_timeout,
        cluster=cluster,
        node_store=node_store,
        max_total_rows=max_total_rows,
        max_continuation_frames=max_continuation_frames,
        trust_server_heartbeat=trust_server_heartbeat,
        close_timeout=close_timeout,
        max_attempts=max_attempts,
        max_elapsed_seconds=max_elapsed_seconds,
        dial_func=dial_func,
        concurrent_leader_conns=concurrent_leader_conns,
        redirect_policy=redirect_policy,
        max_message_size=max_message_size,
    )
    await pool.initialize()
    return pool
