"""Async Python client for dqlite.

Connections and pools are NOT thread-safe and must be used within a single
event loop; submit cross-thread work via ``asyncio.run_coroutine_threadsafe()``.
Free-threaded Python (no-GIL) is unsupported (guarded in ``dqlitewire.__init__``).
"""

import asyncio
import contextlib
import logging
from collections.abc import Sequence as _Sequence
from typing import Final as _Final

from dqliteclient._dial import DialFunc
from dqliteclient.cluster import (
    ClusterClient,
    LeaderInfo,
    NodeMetadata,
    RedirectPolicy,
    allowlist_policy,
    default_safe_redirect_policy,
)
from dqliteclient.connection import CLOSE_TIMEOUT_FLOOR as _CLOSE_TIMEOUT_FLOOR
from dqliteclient.connection import CLOSE_TIMEOUT_FLOOR_RATIONALE as _CLOSE_TIMEOUT_FLOOR_RATIONALE
from dqliteclient.connection import DEFAULT_CLOSE_TIMEOUT_SECONDS as _DEFAULT_CLOSE_TIMEOUT_SECONDS
from dqliteclient.connection import DEFAULT_TIMEOUT_SECONDS as _DEFAULT_TIMEOUT_SECONDS
from dqliteclient.connection import (
    DqliteConnection,
    get_current_pid,
    parse_address,
    validate_timeout,
)
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
from dqliteclient.protocol import DEFAULT_MAX_MESSAGE_SIZE as _DEFAULT_MAX_MESSAGE_SIZE
from dqliteclient.protocol import validate_positive_int_or_none
from dqliteclient.retry import retry_with_backoff
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)

# ``Final`` does not propagate through ``from X import Y`` aliases; the
# re-export is a new binding needing its own annotation, hence the re-pin.
CLOSE_TIMEOUT_FLOOR: _Final[float] = _CLOSE_TIMEOUT_FLOOR
CLOSE_TIMEOUT_FLOOR_RATIONALE: _Final[str] = _CLOSE_TIMEOUT_FLOOR_RATIONALE
DEFAULT_CLOSE_TIMEOUT_SECONDS: _Final[float] = _DEFAULT_CLOSE_TIMEOUT_SECONDS
DEFAULT_TIMEOUT_SECONDS: _Final[float] = _DEFAULT_TIMEOUT_SECONDS
DEFAULT_MAX_MESSAGE_SIZE: _Final[int] = _DEFAULT_MAX_MESSAGE_SIZE

__version__: _Final[str] = "0.3.0"

logger = logging.getLogger(__name__)
# NullHandler suppresses the lastResort stderr emission for apps that have
# not configured logging (Python logging HOWTO convention).
logger.addHandler(logging.NullHandler())

__all__ = [
    "CLOSE_TIMEOUT_FLOOR",
    "CLOSE_TIMEOUT_FLOOR_RATIONALE",
    "DEFAULT_CLOSE_TIMEOUT_SECONDS",
    "DEFAULT_MAX_MESSAGE_SIZE",
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
    "get_current_pid",
    "parse_address",
    "retry_with_backoff",
    "validate_positive_int_or_none",
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
    """Connect to a dqlite node.

    ``timeout`` is per-RPC-phase, not end-to-end; wrap in ``asyncio.timeout(...)``
    for a total deadline. A supplied ``dial_func`` owns all socket options,
    bypassing the default SO_KEEPALIVE/happy-eyeballs setup.
    """
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
        # Clean up the partially-built connection so loop-bound primitives
        # are not left orphaned until GC. The shield lets close() finish even
        # if a fresh outer cancel lands mid-await, so the bare ``raise`` below
        # re-delivers the ORIGINAL connect error rather than a CancelledError.
        # The explicit Task + observer prevents an orphaned shield-created Task
        # from logging "Task exception was never retrieved" at GC.
        from dqliteclient.cluster import _observe_drain_exception

        inner_drain = asyncio.ensure_future(conn.close())
        inner_drain.add_done_callback(_observe_drain_exception)
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(inner_drain)
        except Exception:
            logger.debug(
                "connect: cleanup-close after failed connect",
                exc_info=True,
            )
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
    """Create a connection pool with automatic leader detection.

    ``min_size`` is a pre-warm count, not a steady-state floor (see
    :class:`ConnectionPool` refill semantics). ``timeout`` is per-RPC-phase,
    not end-to-end. ``addresses`` is validated inline on the loop, which can
    be costly near the 10_000-entry cap. ``dial_func`` is mutually exclusive
    with ``cluster=`` (raises ``ValueError``), which carries its own dialer.
    """
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
