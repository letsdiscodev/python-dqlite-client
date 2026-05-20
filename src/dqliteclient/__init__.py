"""Async Python client for dqlite.

Thread safety: connections and pools are NOT thread-safe. All operations
must be performed within a single asyncio event loop. To submit work from
other threads, use ``asyncio.run_coroutine_threadsafe()``. Free-threaded
Python (no-GIL) is not supported — the guard lives in
``dqlitewire.__init__`` (an unconditional transitive dependency via the
``from dqliteclient.connection import DqliteConnection`` chain) and
raises ``ImportError`` at import time.
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
from dqliteclient.connection import (
    CLOSE_TIMEOUT_FLOOR_RATIONALE,
    DqliteConnection,
    get_current_pid,
    parse_address,
    validate_timeout,
)
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
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, NodeStore, YamlNodeStore
from dqliteclient.pool import ConnectionPool
from dqliteclient.protocol import validate_positive_int_or_none
from dqliteclient.retry import retry_with_backoff
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)

__version__: _Final[str] = "0.1.6"

logger = logging.getLogger(__name__)
# Convention from the Python logging HOWTO: attach a ``NullHandler``
# to the library's top-level logger so applications that have not
# configured logging don't see the ``lastResort`` stderr emission,
# and downstream code can silence the library cleanly via
# ``getLogger("dqliteclient").propagate = False``.
logger.addHandler(logging.NullHandler())

__all__ = [
    "CLOSE_TIMEOUT_FLOOR_RATIONALE",
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
    timeout: float = 10.0,
    dial_timeout: float | None = None,
    attempt_timeout: float | None = None,
    max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
    max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
    trust_server_heartbeat: bool = False,
    close_timeout: float = 0.5,
    dial_func: DialFunc | None = None,
) -> DqliteConnection:
    """Connect to a dqlite node.

    Args:
        address: Node address in "host:port" format
        database: Database name to open
        timeout: Per-RPC-phase timeout in seconds (forwarded). Each
            phase of an operation gets the full budget independently;
            wrap with ``asyncio.timeout(...)`` to enforce an end-to-end
            deadline.
        dial_timeout: Per-dial TCP-establish budget. Defaults to
            ``timeout`` when ``None``. Mirrors go-dqlite's
            ``Config.DialTimeout``.
        attempt_timeout: Per-attempt envelope wrapping
            dial + handshake + ``open_database``. Defaults to
            ``timeout`` when ``None``. Mirrors go-dqlite's
            ``Config.AttemptTimeout``.
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
        dial_func: Optional caller-supplied dialer
            (:data:`DialFunc`) replacing the default TCP path. When
            supplied, the default helper's SO_KEEPALIVE / TCP keepalive
            tunables / happy-eyeballs are bypassed — the caller's
            dialer owns all socket options. ``None`` preserves
            existing behaviour. Mirrors go-dqlite's ``WithDialFunc``.

    Returns:
        A connected DqliteConnection
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
    )
    try:
        await conn.connect()
    except BaseException:
        # Eager-connect failure: clean up the partially-constructed
        # connection so loop-bound primitives (locks, transport
        # handles, reader Task) are not left referenced only by the
        # orphan ``conn`` until GC. Mirrors
        # ``dqlitedbapi.aio.aconnect`` and
        # ``sqlalchemydqlite.aio.DqliteDialect_aio.connect``.
        #
        # ``asyncio.shield`` lets the inner ``close()`` task run to
        # completion even when a FRESH outer cancel (e.g. from an
        # ``asyncio.timeout(...)`` wrapping the caller's
        # ``await connect(...)``) lands while we are suspended in
        # ``await conn.close()``. Without the shield, the close
        # would be cancelled mid-flight and the bare ``raise`` below
        # would re-raise a ``CancelledError`` from the close site
        # instead of the original connect-time exception — the
        # original would survive only as ``__context__``.
        # ``contextlib.suppress(asyncio.CancelledError)`` absorbs
        # the outer-await CancelledError so the bare ``raise``
        # below re-delivers the ORIGINAL exception (asyncio will
        # re-raise the cancel at the next await on this task).
        # ``except Exception`` catches non-cancel close-time
        # failures (e.g. OSError on a stale transport) and logs
        # them at DEBUG so the original connect error remains
        # user-visible. Mirrors the ``ClusterClient.connect``
        # try_connect cleanup arm and the ``dqlitedbapi.aio.aconnect``
        # cleanup-close shield.
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(conn.close())
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
    timeout: float = 10.0,
    dial_timeout: float | None = None,
    attempt_timeout: float | None = None,
    cluster: ClusterClient | None = None,
    node_store: NodeStore | None = None,
    max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
    max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
    trust_server_heartbeat: bool = False,
    close_timeout: float = 0.5,
    max_attempts: int | None = None,
    max_elapsed_seconds: float | None = None,
    dial_func: DialFunc | None = None,
    concurrent_leader_conns: int | None = None,
    redirect_policy: RedirectPolicy | None = None,
) -> ConnectionPool:
    """Create a connection pool with automatic leader detection.

    Args:
        addresses: List of node addresses in "host:port" format. Ignored if
            ``cluster`` or ``node_store`` is provided.
        database: Database name to open
        min_size: Number of connections to pre-warm at
            :meth:`ConnectionPool.initialize`. NOT a steady-state
            floor — see :class:`ConnectionPool` for the post-cascade
            refill semantics.
        max_size: Maximum number of connections
        timeout: Per-RPC-phase timeout in seconds (forwarded). Each
            phase of an operation gets the full budget independently;
            wrap with ``asyncio.timeout(...)`` to enforce an end-to-end
            deadline.
        dial_timeout: Per-dial TCP-establish budget. Defaults to
            ``timeout`` when ``None``. Mirrors go-dqlite's
            ``Config.DialTimeout``. Forwarded to the underlying
            ``ClusterClient`` and ``DqliteConnection`` instances.
        attempt_timeout: Per-attempt envelope wrapping dial +
            handshake + ``open_database``. Defaults to ``timeout``
            when ``None``. Mirrors go-dqlite's
            ``Config.AttemptTimeout``.
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
        max_elapsed_seconds: Total wall-clock cap on the per-connect
            retry loop (forwarded to ``ClusterClient.connect``).
            ``None`` (default) means only ``max_attempts`` governs
            termination. Set to a positive finite number for
            go-dqlite-style total-time bounding.
        dial_func: Optional caller-supplied dialer
            (:data:`DialFunc`) forwarded to the auto-built
            :class:`ClusterClient` and to every pooled
            :class:`DqliteConnection`. Mutually exclusive with
            ``cluster=``: an externally-owned cluster already carries
            its own ``dial_func``, so supplying both raises
            ``ValueError``. Mirrors go-dqlite's ``WithDialFunc``.

    Returns:
        An initialized ConnectionPool
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
    )
    await pool.initialize()
    return pool
