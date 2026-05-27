"""Cluster management and leader detection for dqlite."""

import asyncio
import contextlib
import logging
import math
import os
import random
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, NoReturn, final

from dqliteclient import connection as _conn_mod
from dqliteclient._dial import DialFunc, open_connection
from dqliteclient.connection import (
    DEFAULT_CLOSE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DqliteConnection,
    parse_address,
    validate_timeout,
)
from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeStore
from dqliteclient.node_store import NodeInfo as _StoreNodeInfo
from dqliteclient.protocol import DqliteProtocol, validate_positive_int_or_none
from dqliteclient.retry import retry_with_backoff
from dqlitewire import (
    DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES,
)
from dqlitewire import (
    DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS,
)

# Aliased from the wire layer to avoid silent shadowing of
# ``dqliteclient.node_store.NodeInfo`` (the public user-facing class
# also re-exported as ``dqliteclient.NodeInfo``). The two classes
# share field shape (node_id, address, role) but are distinct types;
# ``isinstance`` against the user-facing class on a ``cluster_info()``
# return value depends on the disambiguation. Wire-side NodeInfo is
# used internally by ``cluster_info()`` to decode ``ServersResponse``;
# the user-facing NodeInfo lives in ``node_store``.
from dqlitewire import NodeInfo as _WireNodeInfo
from dqlitewire import NodeRole, sanitize_for_log
from dqlitewire import sanitize_server_text as _sanitize_display_text

__all__ = [
    "ClusterClient",
    "LeaderInfo",
    "NodeMetadata",
    "RedirectPolicy",
    "allowlist_policy",
    "default_safe_redirect_policy",
]

logger = logging.getLogger(__name__)

# Type alias for a redirect-target policy. Returns True if the address
# should be accepted, False to reject with a ClusterError.
#
# The callable is invoked **synchronously** from inside the
# leader-probe loop, holding the per-sweep semaphore. It MUST be cheap
# and non-blocking: no socket I/O, no DNS resolution, no LDAP /
# directory lookup. A slow policy serialises every probe through the
# bound (defeating the parallel sweep) and blocks the asyncio event
# loop, stalling other unrelated tasks. For in-process / in-memory
# allowlist checks (``allowlist_policy(...)`` or a
# ``frozenset.__contains__`` lookup) the synchronous shape is right.
# A future async variant would arrive as a separate type alias if a
# real use case appears.
type RedirectPolicy = Callable[[str], bool]

# Default attempt count for connect(). Three attempts cover one leader
# change plus one transport hiccup; substantially higher counts risk
# hiding genuine cluster instability under what looks like "a slow
# connect". Operators can override via ClusterClient.connect(
# max_attempts=...).
_DEFAULT_CONNECT_MAX_ATTEMPTS: Final[int] = 3

# Per-iteration backoff cap for ``ClusterClient.connect``'s retry
# loop. Matches go-dqlite's ``Config.BackoffCap`` default of 1 s
# (see ``internal/protocol/config.go``). Lower than ``retry.py``'s
# global ``max_delay=10.0`` because leader discovery has a tighter
# latency SLO than a generic retry path: under sustained leader
# churn, a 10 s ceiling means a 6th attempt waits 10 s before
# trying again, whereas 1 s keeps pacing aligned with go-dqlite's
# observed behaviour. Other ``retry_with_backoff`` callers keep
# the 10 s default.
_DEFAULT_CONNECT_MAX_DELAY: Final[float] = 1.0

# Default cap on simultaneous in-flight leader probes during a single
# ``find_leader`` sweep. Mirrors go-dqlite's
# ``Config.ConcurrentLeaderConns`` (default 10, see
# ``internal/protocol/config.go``). Bounds outbound TCP dials per sweep
# so a 500-node cluster does not open 500 sockets at once. Operators
# can override via ``ClusterClient(concurrent_leader_conns=...)``.
_DEFAULT_CONCURRENT_LEADER_CONNS: Final[int] = 10

# Cap per-node error messages at this length before concatenating them
# into the final ClusterError. A failing peer that returns a multi-MB
# FailureResponse message would otherwise produce an O(N * M) string
# held in memory and serialised into every traceback.
_MAX_ERROR_MESSAGE_SNIPPET: Final[int] = 200

# Cap the aggregate-of-all-per-node-errors payload before raising the
# final ``ClusterError``. The per-node cap above bounds the M axis, but
# the N axis (configured node-store size) is still operator-controlled
# and unbounded — a 500-node store all returning hostile-cap-sized
# messages produces ~100 KB of error text held in the ClusterError's
# args, in every traceback render, and in every ``__cause__`` walk.
# 16 KiB / 200 codepoints/snippet ≈ 80 nodes' worth of detail before
# truncation, which is enough for diagnostic utility on any realistic
# cluster while keeping the exception payload bounded.
_MAX_AGGREGATE_ERROR_PAYLOAD: Final[int] = 16 * 1024

# Cap on the number of child exceptions packed into a
# ``BaseExceptionGroup`` constructed from per-node failures. The
# joined display string is already capped (see above); this cap
# bounds the chain length itself so a 500-node hostile-cap-sized
# cluster cannot produce a multi-megabyte exception graph that
# survives cross-process pickling (Celery / multiprocessing pool /
# structured-error capture). 20 is the knee for typical cluster
# sizes (3-9 nodes); the cap fires only on unusually large clusters
# where per-node detail is mostly redundant noise anyway.
_MAX_AGGREGATE_CHILDREN: Final[int] = 20

# How many ``_probe_one`` tasks the find-leader fan-out creates
# before ceding one scheduler tick to siblings. The wire cap
# ``_MAX_NODE_COUNT`` (10_000 in ``dqlitewire.messages.responses``)
# bounds the upper end of the NodeStore size; on a Pi-class device
# allocating that many tasks back-to-back without yielding costs
# tens of milliseconds of pure loop CPU. Yielding every 256 keeps
# the burst well under one frame at 60 fps while preserving the
# all-tasks-created-up-front semantics that the parallel
# verify-redirect optimisation depends on (see ``_probe_one``
# semaphore-scope comment).
_PROBE_TASK_CREATE_YIELD_EVERY: Final[int] = 256


def _bounded_group(message: str, excs: list[BaseException]) -> BaseExceptionGroup[BaseException]:
    """Build a ``BaseExceptionGroup`` with at most
    ``_MAX_AGGREGATE_CHILDREN`` children. Excess children are
    summarised by a synthetic ``DqliteError`` so the operator sees
    "and N more" without losing the chain entirely.

    Used by every site that constructs a per-node aggregate
    exception graph (find_leader, pool.initialize, retry exhaustion)
    so the chain length itself is bounded — the joined display string
    is capped separately by ``_MAX_AGGREGATE_ERROR_PAYLOAD``.
    """
    # Local import to avoid the module-level cluster -> exceptions
    # cycle (exceptions.py is imported by cluster.py at module load
    # time; the synthetic class here is constructed at call time
    # which is safe).
    from dqliteclient.exceptions import DqliteError

    if len(excs) <= _MAX_AGGREGATE_CHILDREN:
        return BaseExceptionGroup(message, excs)
    kept = excs[:_MAX_AGGREGATE_CHILDREN]
    overflow = DqliteError(
        f"... and {len(excs) - _MAX_AGGREGATE_CHILDREN} more per-node failures "
        "(aggregate truncated to keep the exception graph picklable)"
    )
    return BaseExceptionGroup(message, [*kept, overflow])


# Use OS-entropy randomness for the per-sweep node shuffle so that the
# stampede-avoidance is not defeated by a downstream call to
# ``random.seed(...)``. Test suites and some libraries seed the global
# PRNG for determinism; if we used ``random.shuffle`` directly, every
# process picking up that seed would produce the same shuffle and pile
# onto the same node. ``SystemRandom`` ignores ``random.seed()``.
_cluster_random: Final[random.Random] = random.SystemRandom()

# Budget for the bounded writer-drain in ``_query_leader``. A
# responsive peer drains FIN/ACK in microseconds; a slow peer must not
# hold up leader discovery. 100 ms is generous for LAN and still
# negligible against the per-probe ``self._timeout`` (typically seconds).
_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS: Final[float] = 0.1


def _addr_equiv(a: str, b: str) -> bool:
    """Compare host:port strings via the canonical ``(host, port)``
    tuple shape produced by :func:`parse_address`.

    Falls back to literal equality for unparseable inputs so a
    malformed string never crashes the comparison. Hostname-vs-IP
    mismatch (``localhost:9001`` vs ``127.0.0.1:9001``) is not
    canonicalised — DNS resolution belongs elsewhere. Note that
    ``parse_address`` rejects unbracketed IPv6 (per the strict-
    validation hardening), so ``[::1]:9001`` and ``::1:9001`` do
    NOT compare equal — the unbracketed form raises ``ValueError``
    and the fallback compares literal strings.
    """
    try:
        return parse_address(a) == parse_address(b)
    except ValueError:
        return a == b


def _truncate_error(message: str) -> str:
    # Report the overflow count (``len - max``) in the suffix to
    # match the wire-layer ``cap_raw_message`` SSOT and the sibling
    # ``_truncate_for_message`` / ``_truncate_for_log`` helpers in
    # dbapi and SA. Reporting total length here would mislead an
    # operator reading ``[truncated, 4096 chars]`` into thinking the
    # original was 4096 long, when ``max + 4096`` is the real value.
    #
    # Compose ``sanitize_server_text`` (display variant — preserves LF
    # / Tab for multi-line server diagnostics, strips control / bidi /
    # invisible codepoints) so server-supplied text does NOT carry log-
    # splitting characters into the consumed ``raw_message`` attribute.
    # Downstream consumers (SA's ``is_disconnect`` substring matcher,
    # operator-side ``logger.error("%s", exc.raw_message)``) still see
    # readable multi-line content but cannot be tricked into emitting
    # forged log lines by a hostile peer's FailureResponse (CWE-117).
    safe = _sanitize_display_text(message)
    if len(safe) <= _MAX_ERROR_MESSAGE_SNIPPET:
        return safe
    overflow = len(safe) - _MAX_ERROR_MESSAGE_SNIPPET
    return safe[:_MAX_ERROR_MESSAGE_SNIPPET] + f"... [truncated, {overflow} chars]"


def _validate_node_id(node_id: object) -> None:
    """Shared client-side validation for ``node_id`` arguments to
    membership-change methods.

    Rejects ``bool`` (which is otherwise an ``int`` subclass), non-int
    types, and ``< 1`` values. Node id ``0`` is the upstream "no node"
    sentinel (``LeaderResponse.node_id == 0`` means "no leader known"),
    so it cannot be a real cluster member; rejecting client-side keeps
    the diagnostic at the call site instead of the server reply.
    """
    if isinstance(node_id, bool) or not isinstance(node_id, int):
        raise TypeError(f"node_id must be int, got {type(node_id).__name__}")
    if node_id < 1:
        raise ValueError(f"node_id must be >= 1, got {node_id}")


def _observe_drain_exception(t: asyncio.Task[None]) -> None:
    """Done-callback that reaps an inner-shield drain task's exception.

    The leader-probe writer-drain finally wraps ``writer.wait_closed()``
    in ``asyncio.shield(asyncio.wait_for(...))``. When the awaiter is
    cancelled mid-shield and the inner ``wait_for`` later fires
    ``TimeoutError`` (or completes with any other exception), no
    coroutine is awaiting the inner task — asyncio's
    task-finalisation logger emits ``"Task exception was never
    retrieved"`` at GC. Calling ``.exception()`` is the canonical
    "I've observed this" signal so the warning never fires. Mirrors
    the ``_clear_slot`` discipline used by the single-flight
    find-leader slot map.

    Suppress narrowed to ``Exception`` so a ``KeyboardInterrupt`` /
    ``SystemExit`` raised by a signal handler at the exact bytecode
    boundary inside ``t.exception()`` still propagates and drives
    the loop shutdown. Matches the project-wide narrow-suppress
    discipline established at the ``_close_impl`` /
    ``cluster-close-after-shielded`` / ``pool-release-shielded`` /
    ``aio-terminate`` siblings.
    """
    if not t.cancelled():
        with contextlib.suppress(Exception):
            t.exception()


@final
@dataclass(frozen=True, slots=True)
class LeaderInfo:
    """``(node_id, address)`` pair returned by :meth:`ClusterClient.leader_info`.

    Distinct from :class:`dqlitewire.messages.responses.NodeInfo`
    because the wire's ``LeaderResponse`` body has no role field —
    only id and address. Mirrors go-dqlite's ``Client.Leader``
    return shape (``*NodeInfo`` with the role left zero).
    """

    node_id: int
    address: str


@final
@dataclass(frozen=True, slots=True)
class NodeMetadata:
    """Per-node failure-domain + weight metadata.

    Returned by :meth:`ClusterClient.describe`. Mirrors go-dqlite's
    ``NodeMetadata`` struct returned by ``Client.Describe``. The two
    values tune cluster-topology decisions: weight biases leader
    election within a failure domain; failure_domain identifies
    which fault-isolation group the node belongs to.
    """

    failure_domain: int
    weight: int


@dataclass(frozen=True, slots=True)
class _LeaderHit:
    # A parallel ``find_leader`` probe successfully resolved a leader
    # address. ``address`` has already been redirect-policy-checked
    # if it differs from the responding node's own address.
    address: str


@dataclass(frozen=True, slots=True)
class _ProbeMiss:
    # A parallel ``find_leader`` probe finished without yielding a
    # leader address. ``message`` is the per-node snippet that joins
    # the aggregate ``ClusterError`` text. ``exc`` carries the
    # underlying exception for ``BaseExceptionGroup`` chaining, or
    # ``None`` for the legitimate ``no-leader-known`` reply (which is
    # not an exceptional outcome on the wire).
    message: str
    exc: BaseException | None


class ClusterClient:
    """Client that discovers the current dqlite leader.

    Probes configured peer addresses in order to locate the current
    leader:

    - :meth:`find_leader` is **single-shot** — probes each node once
      in order, returns the first successful leader address, raises
      ``ClusterError`` if every node fails. Appropriate for callers
      that want the raw failure surface of a single leader-probe
      pass.

    - :meth:`connect` wraps leader discovery + connection in
      :func:`dqliteclient.retry.retry_with_backoff` (bounded
      exponential backoff). Use this when the caller wants the
      client to retry on transient failures.

    ``ClusterClient`` holds no long-lived resources — each probe
    opens a short-lived socket — so there is nothing to ``close``.
    The caller owns the :class:`NodeStore` lifetime.
    """

    def __init__(
        self,
        node_store: NodeStore,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        concurrent_leader_conns: int = _DEFAULT_CONCURRENT_LEADER_CONNS,
        redirect_policy: RedirectPolicy | None = None,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        max_message_size: int | None = None,
        trust_server_heartbeat: bool = False,
        dial_func: DialFunc | None = None,
    ) -> None:
        """Initialize cluster client.

        Args:
            node_store: Store for cluster node information
            timeout: Connection timeout in seconds. Acts as the default
                for ``dial_timeout`` and ``attempt_timeout`` when those
                are not explicitly set, and as the per-RPC budget on
                the underlying :class:`DqliteProtocol`.
            dial_timeout: Per-dial TCP-establish budget. Defaults to
                ``timeout`` when ``None``. Mirrors go-dqlite's
                ``Config.DialTimeout`` (5 s in go-dqlite's config). Must
                be ``> 0`` when set explicitly.
            attempt_timeout: Per-attempt envelope (dial + handshake +
                first useful round-trip). Defaults to ``timeout`` when
                ``None``. Mirrors go-dqlite's ``Config.AttemptTimeout``
                (15 s). The attempt envelope nests the dial inside it,
                matching ``connector.go:357-360``. Must be ``> 0``
                when set explicitly.
            concurrent_leader_conns: Maximum number of in-flight leader
                probes during a single ``find_leader`` sweep. Mirrors
                go-dqlite's ``Config.ConcurrentLeaderConns`` (default
                10). Bounds simultaneous outbound TCP dials per sweep
                so a 500-node cluster does not open 500 sockets at
                once. Must be ``>= 1``; ``True``/``False`` are
                rejected to avoid silent coercion to ``1`` / ``0``.
            redirect_policy: Optional callable ``(address) -> bool`` that
                authorizes each leader redirect target. If None (default),
                redirects are accepted — this preserves backward
                compatibility but permits a compromised peer to redirect
                clients to arbitrary hosts (SSRF-style). Supply a
                callable or the :func:`allowlist_policy` helper to
                constrain where redirects can go.
            dial_func: Optional caller-supplied dialer
                (:data:`dqliteclient.DialFunc`) replacing the default
                TCP path on every dial site (leader probes, admin
                connections, per-connection :meth:`connect`). When
                supplied the default helper's SO_KEEPALIVE / TCP
                keepalive tunables / happy-eyeballs are bypassed —
                the caller's dialer owns all socket options. ``None``
                preserves existing behaviour. Mirrors go-dqlite's
                ``WithDialFunc``. Forwarded to every
                :class:`DqliteConnection` this cluster builds.
            max_total_rows: Cumulative row-count cap forwarded to
                every admin path and leader probe via the underlying
                :class:`DqliteProtocol`. ``None`` disables the cap
                (matching go-dqlite's unbounded behaviour). Default
                is :data:`dqlitewire.DEFAULT_MAX_TOTAL_ROWS`
                (10_000_000). Bounds the total rows a single query
                can return across all continuation frames.
            max_continuation_frames: Cap on the number of ROWS
                continuation frames a single query can produce.
                ``None`` disables the cap. Default is
                :data:`dqlitewire.DEFAULT_MAX_CONTINUATION_FRAMES`
                (100_000). Bounds the per-query frame count
                independently of ``max_total_rows`` so a malicious /
                buggy server cannot stream forever in tiny chunks.
            trust_server_heartbeat: When ``True``, the WelcomeResponse's
                advertised heartbeat timeout (in seconds) widens the
                per-connection ``_read_timeout`` after handshake — up
                to a 300s cap. Default ``False`` (server's heartbeat
                value is read but has no effect — go-dqlite advertises
                the timeout but its client does not adjust read
                deadlines on it either; the dqlite C server's
                heartbeat is independent of the client's read
                timeout). Opt-in only when the cluster is
                operator-controlled and a longer read-timeout is
                needed for slow consensus-write round-trips.
        """
        validate_timeout(timeout)
        if dial_timeout is not None:
            validate_timeout(dial_timeout, name="dial_timeout")
        if attempt_timeout is not None:
            validate_timeout(attempt_timeout, name="attempt_timeout")
        if isinstance(concurrent_leader_conns, bool) or not isinstance(
            concurrent_leader_conns, int
        ):
            raise TypeError(
                f"concurrent_leader_conns must be int, got {type(concurrent_leader_conns).__name__}"
            )
        if concurrent_leader_conns < 1:
            raise ValueError(f"concurrent_leader_conns must be >= 1, got {concurrent_leader_conns}")
        self._node_store = node_store
        self._timeout = timeout
        # Default to ``timeout`` so existing callers see no change. The
        # split exists so operators can set a tight TCP-dial budget
        # (drop unreachable peers fast) and a generous attempt
        # envelope (tolerate slow handshake on a healthy but loaded
        # peer). Mirrors go-dqlite's ``DialTimeout`` /
        # ``AttemptTimeout`` semantic.
        self._dial_timeout = dial_timeout if dial_timeout is not None else timeout
        self._attempt_timeout = attempt_timeout if attempt_timeout is not None else timeout
        self._concurrent_leader_conns = concurrent_leader_conns
        self._redirect_policy = redirect_policy
        # DoS / heartbeat governors forwarded to the underlying
        # ``DqliteProtocol`` for every admin path
        # (``open_admin_connection``) and leader probe
        # (``_query_leader``). These were silently bypassed for admin
        # operations before — every cluster_info / dump / etc. ran
        # with the dqlitewire defaults regardless of operator config.
        # ``trust_server_heartbeat`` is security-relevant (opt-in to
        # widened per-read deadline); ``max_total_rows`` and
        # ``max_continuation_frames`` matter for ``dump`` paths
        # carrying multi-GB results.
        #
        # Validate at construction (rather than deferring to
        # ``DqliteProtocol.__init__`` inside ``_query_leader``) so a
        # misconfigured value surfaces at the operator's config-load
        # site instead of as a per-node probe failure deep inside the
        # first ``find_leader()`` aggregate. Mirrors the discipline at
        # ``DqliteConnection.__init__`` and the project-canonical
        # "fail-fast construction" precedent documented at
        # ``connection.py``'s ``parse_address`` call.
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        # Symmetric inbound + outbound frame cap forwarded to every
        # admin path (``open_admin_connection``) and leader probe
        # (``_query_leader``). ``None`` defers to the wire-layer
        # default (64 MiB) via ``DqliteProtocol.__init__``. A prior
        # parity fix extended this class to forward
        # ``max_total_rows`` / ``max_continuation_frames`` /
        # ``trust_server_heartbeat``; ``max_message_size`` was added
        # to the pool/dbapi/SA surface later and was not retrofitted
        # here, so an operator tightening the cap cluster-wide as a
        # DoS hardening measure saw the cap silently bypassed on
        # the admin path (notably ``dump``, where a multi-GB
        # database arrives as one frame per file content).
        #
        # Validate at construction mirroring the shape at
        # ``DqliteProtocol.__init__``: ``None`` defers to the wire-
        # layer default; any int must be >= 1; ``bool`` rejected
        # (PEP-484-style — ``True`` is technically int but operators
        # mean a count, not a flag).
        if max_message_size is not None:
            if isinstance(max_message_size, bool) or not isinstance(max_message_size, int):
                raise TypeError(
                    f"max_message_size must be int or None, got {type(max_message_size).__name__}"
                )
            if max_message_size < 1:
                raise ValueError(f"max_message_size must be >= 1, got {max_message_size}")
        self._max_message_size = max_message_size
        self._trust_server_heartbeat = trust_server_heartbeat
        self._dial_func: DialFunc | None = dial_func
        # Last-known-leader cache (mirror of go-dqlite's
        # ``LeaderTracker.lastKnownLeaderAddr``). On every successful
        # sweep we set this; on the next ``find_leader`` we probe it
        # directly first and fall through to the full sweep on miss.
        # CPython-atomic attribute write/read — the GIL serialises a
        # single ``str | None`` assignment, so updates are visible
        # across coroutines without a lock. A future port to free-
        # threaded Python (PEP 703) MUST add a lock or atomic op
        # around get/set; do not extend the contents of this
        # attribute (e.g. into a ``(addr, expires_at)`` tuple) without
        # first introducing such a lock.
        self._last_known_leader: str | None = None
        # Single-flight slot map for ``find_leader``. Concurrent callers
        # share the in-flight discovery task instead of each launching
        # an independent per-node sweep — under a leader flip with N
        # waiting acquirers, this collapses N×M handshake attempts
        # against the failing ex-leader into M (one sweep). Slots are
        # cleared on done-callback so a fresh probe re-runs after the
        # current task finishes; consecutive callers do NOT share a
        # cached failure.
        #
        # The map is keyed by ``(trust_server_heartbeat,)``: callers
        # that requested different heartbeat-trust semantics must NOT
        # collapse onto the same task — the per-call flag would
        # otherwise be silently ignored for the second caller, which
        # is a security-adjacent regression for operators who opted
        # one pool out of widened heartbeats. The slot key also
        # includes the per-call ``policy`` override so callers passing
        # distinct audit-mode policies do not share each other's
        # results.
        self._find_leader_tasks: dict[tuple[bool, RedirectPolicy | None], asyncio.Task[str]] = {}
        # Fork-after-init: the slot map holds asyncio.Task instances
        # bound to the parent's loop. A child that forks mid-sweep
        # would observe an inherited task and ``await
        # asyncio.shield(<parent-loop task>)`` — undefined behaviour.
        # Symmetric with the DqliteConnection / ConnectionPool
        # pid guards.
        self._creator_pid = os.getpid()

    def _check_pid(self) -> None:
        """Raise ``InterfaceError`` if called in a forked child.

        Every public admin RPC routes through here so the per-address
        escape hatches (``describe(address=...)``,
        ``set_weight(address=...)``, ``open_admin_connection``) carry
        the same fork-after-init discipline as the leader-routed
        siblings. Bypassing this gate would leave the instance
        "alive enough" to describe a specific node but "dead" for
        leader-routed calls, surfacing as confusing
        ``InterfaceError`` from one method and successful
        operation from the next.
        """
        if _conn_mod.get_current_pid() != self._creator_pid:
            raise InterfaceError(
                f"ClusterClient used after fork; reconstruct from "
                f"configuration in the target process. (created in "
                f"pid {self._creator_pid}, current pid {_conn_mod.get_current_pid()})"
            )

    @classmethod
    def from_addresses(
        cls,
        addresses: Sequence[str],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        concurrent_leader_conns: int = _DEFAULT_CONCURRENT_LEADER_CONNS,
        redirect_policy: RedirectPolicy | None = None,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        max_message_size: int | None = None,
        trust_server_heartbeat: bool = False,
        dial_func: DialFunc | None = None,
    ) -> "ClusterClient":
        """Create cluster client from list of addresses.

        ``dial_timeout`` and ``attempt_timeout`` mirror go-dqlite's
        ``DialTimeout`` / ``AttemptTimeout`` configuration knobs and
        default to ``timeout`` when ``None``.

        ``concurrent_leader_conns`` (default 10) bounds the number of
        in-flight leader probes during a single ``find_leader`` sweep
        and is forwarded to the underlying ``ClusterClient``. Mirrors
        go-dqlite's ``Config.ConcurrentLeaderConns`` semantic. Note
        that ``create_pool`` does not currently surface this knob —
        operators wanting a non-default value can construct a
        ``ClusterClient`` directly and pass it via ``cluster=``.
        """
        store = MemoryNodeStore(addresses)
        return cls(
            store,
            timeout=timeout,
            dial_timeout=dial_timeout,
            attempt_timeout=attempt_timeout,
            concurrent_leader_conns=concurrent_leader_conns,
            redirect_policy=redirect_policy,
            max_total_rows=max_total_rows,
            max_continuation_frames=max_continuation_frames,
            max_message_size=max_message_size,
            trust_server_heartbeat=trust_server_heartbeat,
            dial_func=dial_func,
        )

    def __reduce__(self) -> NoReturn:
        # Holds a per-client single-flight slot map keyed by loop-bound
        # asyncio.Task instances and a NodeStore that may itself hold
        # mutable address state. Pickling produces a duplicate detached
        # from any loop and from the original NodeStore's lifecycle —
        # any use yields opaque corruption. Surface a clear
        # driver-level TypeError instead. Symmetric with the
        # ConnectionPool / DqliteConnection guards.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — holds a "
            f"loop-bound single-flight slot map and a NodeStore "
            f"reference; reconstruct from configuration in the target "
            f"process instead."
        )

    def _get_last_known_leader(self) -> str | None:
        """Return the cached last-known-leader address, or ``None``.

        Single attribute read; CPython-atomic. See ``__init__`` for
        the free-threaded-Python caveat.
        """
        return self._last_known_leader

    def _set_last_known_leader(self, address: str | None) -> None:
        """Update the last-known-leader cache. ``None`` clears it.

        Single attribute write; CPython-atomic. See ``__init__``.
        """
        self._last_known_leader = address

    def _check_redirect(self, address: str, *, policy: RedirectPolicy | None = None) -> None:
        """Reject leader-redirect targets that fail the configured policy.

        ``policy`` overrides the instance-level ``redirect_policy`` when
        provided — used by :meth:`leader_info` to apply a per-call
        policy without mutating the instance default. ``None`` falls
        back to ``self._redirect_policy`` (the standard probe-time
        check).
        """
        effective = policy if policy is not None else self._redirect_policy
        if effective is None:
            return
        if not effective(address):
            # Security-adjacent event: an operator-supplied policy
            # just rejected a server-advised leader target. Surface
            # at DEBUG so SSRF-style attempts or policy
            # misconfigurations are traceable from logs alone,
            # not only through an exception stack.
            logger.debug("cluster: redirect rejected by policy to=%s", sanitize_for_log(address))
            # Wrap the address through ``_sanitize_display_text`` for the
            # exception message — symmetric with the sibling
            # ``_query_leader`` malformed-redirect arms which use the same
            # helper. ``{x!r}`` is preserved on top for visual clarity
            # (quote-delimited so an operator can tell where the
            # address ends). ``raw_message=`` carries the verbatim
            # peer-supplied string so cross-process forensic recovery
            # (Celery worker / result backend / SIEM correlation)
            # captures the un-substituted original, capped at
            # ``_MAX_RAW_MESSAGE`` by ``DqliteError.__init__``.
            raise ClusterPolicyError(
                f"Leader redirect to {_sanitize_display_text(address)!r} "
                f"rejected by redirect_policy",
                raw_message=address,
            )

    async def find_leader(
        self,
        *,
        trust_server_heartbeat: bool = False,
        policy: RedirectPolicy | None = None,
    ) -> str:
        """Find the current cluster leader.

        Returns the leader address. ``trust_server_heartbeat`` is forwarded
        to each probe protocol's constructor for API parity with the
        full-connect path (:class:`DqliteConnection`), but **has no effect
        on the probe traversal**: ``_query_leader`` uses
        :meth:`DqliteProtocol.negotiate_protocol_only` (not the full
        :meth:`DqliteProtocol.handshake`) to skip per-peer client-id
        registration, which means no ``WelcomeResponse`` is read on the
        probe path — so the welcome-driven read-timeout widening branch
        that consumes ``self._trust_server_heartbeat`` never fires. The
        probe read deadline is therefore bounded by ``self._timeout``
        regardless of this flag. Operators tuning ``trust_server_heartbeat``
        for slow-leader-probe scenarios will not see the widening apply to
        the probe ``get_leader`` reads — it only applies to the main
        query path after a full handshake.

        ``policy`` is an optional :data:`RedirectPolicy` callable applied
        to every leader-redirect target encountered during the sweep
        (cached fast path AND parallel arms). Mirrors the precedence
        used by :meth:`leader_info` and :meth:`cluster_info`: falls back
        to ``self._redirect_policy`` when ``None``. Use to enforce a
        strict one-off policy (audit mode) without mutating the
        instance-level default and racing concurrent callers.

        Concurrent callers share an in-flight discovery task (single-
        flight). Under a leader flip with N waiting acquirers, this
        collapses N independent per-node sweeps into one. Failures are
        not cached: once the current task completes, the slot clears
        so the next caller runs a fresh probe.

        Single-flight caveat re ``policy``: the slot key includes the
        policy callable identity (Python's ``==`` on bare functions /
        lambdas falls back to ``is``). Two concurrent callers passing
        distinct policy callables — even semantically-identical
        callables, e.g. two ``lambda a: True`` instances — hash to
        different slot keys and run independent sweeps. The collapse
        works when:

        * Both pass ``None`` (resolves to ``self._redirect_policy`` —
          a single identity at construction time). This is the
          common case for production callers.
        * Both reuse the same module-level policy constant.
        * Both pass a hashable callable that compares equal (e.g. a
          frozen class instance with custom ``__eq__`` / ``__hash__``).

        **The slot does NOT collapse when each caller constructs a
        fresh inline lambda / partial / closure for their probe.**
        SA-engine acquisition loops and dbapi cursor open paths that
        wire a per-call policy this way pay N independent sweeps under
        leader flip — the opposite of the single-flight intent. To get
        single-flight in audit-mode scenarios, prefer constructing the
        ``ClusterClient`` with ``redirect_policy=<your policy>`` and
        let callers pass ``policy=None``; the slot then collapses
        normally. The ``policy=`` per-call kwarg is an opt-OUT path
        for callers that genuinely need isolation per audit pass.

        Single-flight staleness window: the in-flight task snapshots
        the node store once at the top of its sweep
        (``_find_leader_impl``). A concurrent ``set_nodes(...)`` update
        that lands AFTER the sweep started is NOT visible to the
        sharing waiters — they see the result computed against the
        pre-update node list. The next sweep picks up the new nodes,
        so the staleness window is bounded by one sweep duration. The
        single-flight collapse is a deliberate trade-off: the alternative
        (re-poll the node store between probes) costs an extra await
        per probe to close a window most callers do not exercise.

        Address-only return: go-dqlite's connector adopts the probe
        socket on success and returns the open ``*Protocol``;
        Python's ``find_leader`` returns just an address string,
        and high-level callers (:meth:`connect`,
        :meth:`ConnectionPool._create_connection`,
        :meth:`open_admin_connection`) re-dial. The cost is one
        extra TCP three-way + handshake RTT per success path on
        the leader — the leader pays N round-trips for N
        concurrent acquirers if no single-flight collapse applies,
        and one round-trip for N collapsed acquirers if it does.
        Operators sizing ``attempt_timeout`` on high-RTT WAN
        deployments should budget for the extra round trip.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard before any transport
                activity; reconstruct the client in the target
                process to recover.
            ClusterError: when no node in the configured node store
                responds with a leader address (every peer either
                refused the connection, timed out, or self-reported
                no leader).
            DqliteConnectionError: when the cached fast-path arm
                attempts a single-RTT probe and the cached peer
                cannot be reached (rewrapped from
                ``TimeoutError`` / ``OSError`` /
                ``ProtocolError``).
            ClusterPolicyError: when a redirect target is rejected
                by ``policy`` / ``self._redirect_policy``.
        """
        self._check_pid()
        # Resolve ``policy=None`` to ``self._redirect_policy`` BEFORE
        # building the slot key. Otherwise callers passing ``None`` and
        # callers passing ``self._redirect_policy`` explicitly hash to
        # different keys despite the same effective behaviour, causing
        # the single-flight collapse to miss and two concurrent probes
        # to run for the same effective policy. Internal re-dispatch
        # paths (e.g. ``_check_redirect`` retries at L1683/L1837) forward
        # ``policy=policy`` explicitly, so the collision is reachable
        # from first-party code, not just a theoretical edge case.
        effective_policy: RedirectPolicy | None = (
            policy if policy is not None else self._redirect_policy
        )
        key: tuple[bool, RedirectPolicy | None] = (trust_server_heartbeat, effective_policy)
        task = self._find_leader_tasks.get(key)
        if task is None or task.done():
            task = asyncio.create_task(
                self._find_leader_impl(
                    trust_server_heartbeat=trust_server_heartbeat,
                    policy=effective_policy,
                )
            )

            def _clear_slot(t: asyncio.Task[str]) -> None:
                # Clear the slot only if it still points at THIS task —
                # a concurrent ``find_leader`` may have already
                # supplanted us if our task finished and a new caller
                # triggered a fresh probe before this callback ran.
                if self._find_leader_tasks.get(key) is t:
                    del self._find_leader_tasks[key]
                # Observe any exception so asyncio's GC does not log
                # "Task exception was never retrieved" when every
                # external ``await asyncio.shield(task)`` caller was
                # cancelled out before the inner task resolved (e.g.
                # an outer ``asyncio.timeout`` fired during a leader
                # discovery cascade). ``task.exception()`` is the
                # canonical "I've observed this" signal — non-raising
                # on a done-not-cancelled task. Narrow the suppress
                # to ``Exception`` so a ``KeyboardInterrupt`` /
                # ``SystemExit`` raised by a signal handler at a
                # bytecode boundary inside the callback still
                # propagates and drives loop shutdown — matches the
                # project-wide narrow-suppress discipline.
                if not t.cancelled():
                    with contextlib.suppress(Exception):
                        t.exception()

            # Register the done-callback BEFORE inserting into the
            # shared slot so a signal-driven interrupt (KI / SystemExit
            # raised by a signal handler at any bytecode boundary)
            # between ``create_task`` and ``add_done_callback`` cannot
            # leave the slot pointing at a task whose completion is
            # never observed. ``add_done_callback`` is safe on a
            # not-yet-started task; the callback fires regardless of
            # slot state, and ``_clear_slot``'s "only delete if it
            # still points at us" guard handles the case where the
            # slot was never set.
            task.add_done_callback(_clear_slot)
            self._find_leader_tasks[key] = task
        # Shield so a caller's outer cancel does not kill the shared
        # task; the cancel still propagates to the calling coroutine
        # via ``await asyncio.shield``.
        return await asyncio.shield(task)

    async def _find_leader_impl(
        self,
        *,
        trust_server_heartbeat: bool,
        policy: RedirectPolicy | None = None,
    ) -> str:
        """Perform the actual leader discovery sweep. See ``find_leader``
        for the public contract — this method is the single-flight
        backing implementation.

        Probes nodes in parallel bounded by
        ``self._concurrent_leader_conns`` (default 10, matching
        go-dqlite's ``Config.ConcurrentLeaderConns``). The first probe
        that resolves to a leader address wins; siblings are
        cancelled and their sockets drained via ``_query_leader``'s
        ``finally:`` 100 ms bounded drain. Mirrors
        ``connector.go::connectAttemptAll``.

        Error-accumulator note: per-node failures (DqliteConnectionError,
        ProtocolError, OperationalError, OSError, TimeoutError) are
        converted to ``_ProbeMiss`` outcomes and captured into the
        ``errors`` / ``per_node_excs`` accumulators so the final
        ``ClusterError`` message names every probed node.
        ``ClusterPolicyError`` raised by ``_check_redirect`` is NOT
        caught inside the per-probe coroutine: it surfaces via
        ``task.exception()`` in the gather loop and propagates after
        cancelling siblings, dropping any accumulated probe history.
        That's intentional: policy rejections are deterministic
        configuration errors, so the same policy would apply to every
        other node and the probe history is not actionable.

        Fast path: if a previous sweep set ``_last_known_leader``,
        probe that address first. On hit, return it; on miss
        (transport error, no-leader reply, address mismatch), clear
        the cache and fall through to the parallel sweep. Mirrors
        ``connector.go::connectAttemptAll`` (lines 228-237). The
        single-flight slot map collapses concurrent ``find_leader``
        callers, so a stale-cache miss costs one extra probe before
        the full sweep — same staleness window as the existing
        single-flight contract.
        """
        cached = self._get_last_known_leader()
        if cached is not None:
            try:
                # Use ``asyncio.timeout`` (cancel-scope semantics)
                # rather than ``asyncio.wait_for`` (which discards
                # the inner-task result on outer-cancel — the
                # canonical leak shape for this class of bug).
                # Mirrors the discipline at ``_query_leader``'s
                # outer-dial scope, ``open_admin_connection``, and
                # ``_connect_impl`` — see those sites for the same
                # rationale.
                async with asyncio.timeout(self._attempt_timeout):
                    cached_leader = await self._query_leader(
                        cached,
                        trust_server_heartbeat=trust_server_heartbeat,
                    )
                if cached_leader:
                    if not _addr_equiv(cached_leader, cached):
                        # Cached node redirected us elsewhere.
                        # ``_check_redirect`` may raise
                        # ``ClusterPolicyError`` — handled below.
                        self._check_redirect(cached_leader, policy=policy)
                        # Re-verify the redirect target self-identifies
                        # as leader before trusting the hint. On
                        # mismatch (stale-hint cached node), clear the
                        # cache and fall through to the full sweep so
                        # leader rediscovery runs. Without this, a
                        # cached responder pointing to an ex-leader
                        # would loop the caller through a wasted
                        # ``connect()``+Open before retry.
                        #
                        # Pass ``attempt_timeout`` rather than the
                        # default ``dial_timeout``: the cached fast-
                        # path is single-RTT followed by a single
                        # verify-RTT — sequential, not nested. The
                        # ``_verify_redirect`` doc-default of
                        # ``dial_timeout`` reflects the parallel-sweep
                        # call site at ``_probe_one``, which runs
                        # SIBLING (not nested) to the initial probe
                        # after the semaphore release. Using
                        # ``dial_timeout`` here would drop healthy-but-
                        # loaded redirect targets when operators size
                        # ``dial_timeout < attempt_timeout``.
                        verified = await self._verify_redirect(
                            cached_leader,
                            trust_server_heartbeat=trust_server_heartbeat,
                            timeout=self._attempt_timeout,
                        )
                        if verified is None:
                            logger.debug(
                                "find_leader: fast-path probe of cached "
                                "leader %s redirected to %s but "
                                "verification failed; clearing cache "
                                "and falling through to full sweep",
                                sanitize_for_log(cached),
                                sanitize_for_log(cached_leader),
                            )
                            self._set_last_known_leader(None)
                            # Skip the no-leader-known log below and
                            # let control reach the full sweep
                            # naturally.
                        else:
                            self._set_last_known_leader(cached_leader)
                            return cached_leader
                    else:
                        # Cached node confirmed itself as leader.
                        # Re-validate against the redirect policy so a
                        # tightened allowlist takes effect without
                        # waiting for the next leader flip. The cache
                        # is already correct for this address; no
                        # write needed.
                        self._check_redirect(cached_leader, policy=policy)
                        return cached_leader
                else:
                    # Cached node replied with no-leader-known: the
                    # leader has flipped or stepped down. Clear the
                    # cache and fall through.
                    logger.debug(
                        "find_leader: fast-path probe of cached leader %s "
                        "returned no-leader-known; clearing cache and falling "
                        "through to full sweep",
                        sanitize_for_log(cached),
                    )
                    self._set_last_known_leader(None)
            except (
                DqliteConnectionError,
                ProtocolError,
                OperationalError,
                OSError,
                TimeoutError,
                ValueError,
            ) as e:
                # Fast-path miss: probe failed. Clear the cache and
                # fall through to the full sweep. Log at DEBUG so
                # operators tailing logs see the cache invalidation
                # without spamming default-verbosity output.
                # ``ValueError`` is included to translate
                # ``parse_address`` rejection when a third-party
                # ``NodeStore`` returns malformed entries (matches the
                # parallel-sweep ``_probe_one`` precedent below).
                logger.debug(
                    "find_leader: fast-path probe of cached leader %s failed (%s); "
                    "clearing cache and falling through to full sweep",
                    sanitize_for_log(cached),
                    type(e).__name__,
                )
                self._set_last_known_leader(None)
            except ClusterPolicyError as policy_exc:
                # The cached address redirected us to a policy-
                # rejected target (or the operator changed the
                # redirect policy). Clear and propagate — the same
                # policy applies to every other probe, so falling
                # through would not produce a different outcome.
                #
                # Preserve the cached-address forensic trail: chain
                # via ``from`` so the raised ``ClusterPolicyError``'s
                # ``__cause__`` records which cached node redirected
                # us to the rejected target. The parallel-sweep arm
                # below (lines 1349-1356) accumulates this history
                # for the sweep path; the fast-path raise unwinds
                # past that accumulator, so without the explicit
                # chain here a forensic walker sees only a bare
                # ``ClusterPolicyError`` with no breadcrumb back to
                # the cached address. ``ClusterError`` here is the
                # cheapest carrier of the "fast-path redirect
                # rejected" diagnostic with the sanitised cached
                # address embedded; the operator's bug-class
                # remains ``ClusterPolicyError`` (the outer raise).
                self._set_last_known_leader(None)
                raise policy_exc from ClusterError(
                    f"find_leader: fast-path probe of cached leader "
                    f"{sanitize_for_log(cached)} redirected to a "
                    f"policy-rejected target ({policy_exc})"
                )

        # Snapshot defensively into a fresh list — the NodeStore Protocol
        # documents that the returned Sequence "will not be mutated
        # after return", but a third-party impl that returns its
        # internal storage and mutates it from a concurrent
        # ``set_nodes()`` would otherwise produce a torn iteration
        # deep inside the parallel sweep, surfacing as an obscure
        # ``IndexError`` from ``enumerate(nodes)`` rather than a
        # clean diagnostic. ``MemoryNodeStore`` and ``YamlNodeStore``
        # already return tuples; the cost is O(N) at typical 3-7
        # node store sizes. The list is mutable so the subsequent
        # shuffle + stable-sort can operate in place.
        nodes = list(await self._safe_node_snapshot())

        if not nodes:
            raise ClusterError("No nodes configured")

        # Shuffle first so repeated callers don't stampede the same node;
        # then stable-sort by role so voters come before non-voters.
        # Standby/spare nodes can never become leader (their LEADER
        # response is always (0, "")), so probing them first wastes RTTs.
        # Strict role-ascending order so STANDBY (1) probes before
        # SPARE (2): standbys participate in heartbeats and are more
        # likely to know the current leader, while spares participate
        # in neither voting nor heartbeats and lag strictly more.
        # Matches go-dqlite's ``connector.go::connectAttempt`` sort
        # discipline (see Go's "standbys are more likely to know who
        # the leader is than spares" rationale).
        #
        # Within-role shuffle is preserved by the stable-sort
        # discipline: ``_cluster_random.shuffle`` randomizes node
        # order across parallel callers (stampede-avoidance), then
        # ``sort`` rearranges only by role-bucket while keeping
        # equal-role nodes in their shuffled positions.
        _cluster_random.shuffle(nodes)
        nodes.sort(key=lambda n: int(n.role))

        total_nodes = len(nodes)
        # Cap simultaneous in-flight probes at
        # ``concurrent_leader_conns`` so a 500-node cluster doesn't
        # open 500 sockets at once. Mirrors go-dqlite's
        # ``Config.ConcurrentLeaderConns`` semantic.
        semaphore = asyncio.Semaphore(self._concurrent_leader_conns)

        errors: list[str] = []
        # Collect every per-node BaseException so the final
        # ``ClusterError`` can chain them all via
        # ``BaseExceptionGroup``. Previously only the LAST iteration's
        # exception was preserved on ``__cause__`` — code that
        # branches on the cause class (e.g. routing security alerts
        # for ``ProtocolError``-from-malformed-redirect) saw a non-
        # deterministic decision based on iteration ordering. Mirrors
        # the discipline already applied to ``ConnectionPool.initialize``.
        per_node_excs: list[BaseException] = []

        async def _probe_one(idx: int, node: _StoreNodeInfo) -> _LeaderHit | _ProbeMiss:
            # Per-node probe coroutine. Returns ``_LeaderHit`` on a
            # successful leader resolution, ``_ProbeMiss`` on transport/
            # protocol failures or no-leader-known replies. Lets
            # ``ClusterPolicyError`` propagate so the gather loop can
            # cancel siblings and re-raise.
            #
            # Semaphore scope: the slot is held only across the
            # initial ``_query_leader``. The shielded 100ms shutdown
            # drain in ``_query_leader.finally`` AND the subsequent
            # ``_verify_redirect`` re-probe run OUTSIDE the slot.
            # Holding the slot across the re-probe would bottleneck
            # on redirect-stampede + leader-flip:
            # ``concurrent_leader_conns=10`` would serialize at 10
            # slots × (attempt_timeout + verify + drain). Note that
            # this means ``concurrent_leader_conns`` bounds initial
            # probes only, not redirect re-verify dials — under a
            # full-cluster redirect stampede the verify fan-out can
            # briefly exceed the slot count.
            sem_acquired = False
            try:
                await semaphore.acquire()
                sem_acquired = True
            except (KeyboardInterrupt, SystemExit):
                # A signal-handler raise (synthetic ``KeyboardInterrupt``,
                # ``SystemExit``, or a cross-thread
                # ``PyErr_SetAsyncExc``) landing on the bytecode
                # boundary between ``acquire()`` returning and the
                # ``sem_acquired = True`` store would otherwise leave
                # the permit decremented while the outer ``finally``'s
                # ``if sem_acquired:`` guard misfires — leaking one
                # permit per occurrence. With
                # ``_concurrent_leader_conns=10`` (default) a handful
                # of these wedge the whole sweep. Release defensively
                # before re-raising; ``ValueError`` is suppressed for
                # the (impossible-in-practice but cheap-to-guard) case
                # where the signal beat the decrement and the release
                # would over-credit. Mirrors the threading-lock
                # discipline at ``dqlitedbapi.connection``'s
                # ``_loop_lock`` acquire arm.
                if not sem_acquired:
                    with contextlib.suppress(ValueError):
                        semaphore.release()
                raise
            try:
                try:
                    # ``async with asyncio.timeout(...)`` (cancel-scope
                    # semantics) rather than ``asyncio.wait_for`` (which
                    # cancels the inner task and discards a result that
                    # arrived on the same scheduling slice as the outer
                    # cancel). Mirrors the fast-path arm above and the
                    # other cancel-scope sibling sites
                    # (``_query_leader`` outer-dial,
                    # ``open_admin_connection``, ``_connect_impl``).
                    async with asyncio.timeout(self._attempt_timeout):
                        leader_address = await self._query_leader(
                            node.address,
                            trust_server_heartbeat=trust_server_heartbeat,
                        )
                except TimeoutError as e:
                    # Use the log-bound sanitiser (escapes LF/TAB) so
                    # the address embedded in ``_ProbeMiss.message``
                    # cannot split a downstream log record when a
                    # caller emits ``ClusterError`` via ``logger.error("%s",
                    # err)``. Defence-in-depth against a dial_func
                    # override or a peer that supplied a malformed
                    # address that bypassed ``parse_address`` (CWE-117).
                    # Mirrors the discipline at the redirect-verify
                    # arm below (lines 1128-1149).
                    _safe_addr = sanitize_for_log(node.address)
                    logger.debug(
                        "find_leader: %s timed out after %.3fs (%d/%d)",
                        sanitize_for_log(node.address),
                        self._attempt_timeout,
                        idx + 1,
                        total_nodes,
                    )
                    return _ProbeMiss(message=f"{_safe_addr}: timed out", exc=e)
                except (
                    DqliteConnectionError,
                    ProtocolError,
                    OperationalError,
                    OSError,
                    ValueError,
                ) as e:
                    # Narrow the catch so programming bugs (TypeError,
                    # KeyError, etc.) propagate directly instead of
                    # being stringified into a retryable ClusterError.
                    # ``ValueError`` is included to translate the
                    # ``parse_address`` rejection that fires when a
                    # peer returns a malformed redirect target — without
                    # it, a single hostile node sabotages the parallel
                    # sweep cluster-wide. The existing
                    # ``ClusterPolicyError``-on-``ValueError`` wrap at
                    # ``try_connect`` covers only the constructor path;
                    # this is the inner-sweep counterpart.
                    logger.debug(
                        "find_leader: %s failed with %s: %s (%d/%d)",
                        sanitize_for_log(node.address),
                        type(e).__name__,
                        sanitize_for_log(_truncate_error(str(e))),
                        idx + 1,
                        total_nodes,
                    )
                    return _ProbeMiss(
                        message=(
                            # ``sanitize_for_log`` escapes LF/TAB so the
                            # address (and the truncated error text) in
                            # ``_ProbeMiss.message`` cannot split a
                            # downstream log record (CWE-117). Mirrors
                            # the redirect-verify arm at lines 1128-1149.
                            f"{sanitize_for_log(node.address)}: "
                            f"{sanitize_for_log(_truncate_error(str(e)))}"
                        ),
                        exc=e,
                    )

                # Initial probe complete: release the semaphore slot
                # before the redirect-verify / no-leader-known log
                # work. The outer ``finally`` is a safety net for any
                # exception path that hasn't already released.
                if sem_acquired:
                    semaphore.release()
                    sem_acquired = False

                if leader_address:
                    # Policy applies to EVERY confirmed-leader address,
                    # not just the redirect-different-from-probe case.
                    # The probed node's own address must pass the same
                    # gate as a redirect target — otherwise an
                    # allowlist-style policy is silently bypassed when
                    # the policy-excluded node happens to be the
                    # current leader (the canonical regional-pin use
                    # case). The cached fast-path (cluster.py:867-869,
                    # 919) already gates on every cached-leader
                    # consultation; this restores the probe-path
                    # symmetry. ``_check_redirect`` is idempotent and
                    # inexpensive (a single callable invocation).
                    # Re-raises ClusterPolicyError on rejection; the
                    # gather loop catches it and propagates.
                    self._check_redirect(leader_address, policy=policy)
                    # Only leader_address values that did NOT come from
                    # node.address itself need authorizing — those are
                    # real redirects. Compare via the canonical
                    # (host, port) tuple so an IPv6 bracketing
                    # difference does not look like a redirect.
                    if not _addr_equiv(leader_address, node.address):
                        # Re-probe the redirect target to confirm it
                        # self-identifies as leader. Stale-hint
                        # peers can hand back a node that no longer
                        # holds leadership; trusting the hint
                        # without re-verification wastes a full
                        # ``connect()`` round-trip and produces a
                        # misleading error chain. Mirrors go-dqlite's
                        # ``connector.go::connectAttemptOne`` redirect
                        # re-probe. The semaphore is intentionally
                        # released before this dial so a redirect
                        # stampede does not bottleneck on the
                        # ``concurrent_leader_conns`` budget.
                        verified = await self._verify_redirect(
                            leader_address,
                            trust_server_heartbeat=trust_server_heartbeat,
                        )
                        if verified is None:
                            # Log-bound sanitiser: ``sanitize_for_log``
                            # escapes LF / Tab so a server-supplied
                            # ``leader_address`` cannot split a logger
                            # record into multiple lines (CWE-117).
                            _safe_addr_log = sanitize_for_log(node.address)
                            _safe_hint_log = sanitize_for_log(leader_address)
                            logger.debug(
                                "find_leader: %s redirected to %s "
                                "but verification failed (stale hint); "
                                "falling through (%d/%d)",
                                _safe_addr_log,
                                _safe_hint_log,
                                idx + 1,
                                total_nodes,
                            )
                            # ``_ProbeMiss.message`` ends up in
                            # ``ClusterError.args[0]`` and is logged via
                            # ``logger.exception`` by upstream callers;
                            # use the log-bound sanitiser here too so a
                            # server-supplied LF in ``leader_address``
                            # cannot split downstream log records
                            # (CWE-117 secondary). Mirrors the
                            # LF-stripping discipline at the sibling
                            # probe-failure arm above.
                            return _ProbeMiss(
                                message=(f"{_safe_addr_log}: stale redirect to {_safe_hint_log}"),
                                exc=None,
                            )
                    return _LeaderHit(address=leader_address)

                # ``_query_leader`` returns ``None`` for the legitimate
                # ``(node_id=0, address="")`` "no leader known yet"
                # reply. Without this branch the ``errors`` list
                # silently stays empty.
                # ``sanitize_for_log`` escapes LF/TAB so the address
                # embedded in ``_ProbeMiss.message`` cannot split a
                # downstream log record when a caller emits the
                # surrounding ``ClusterError`` via ``logger.*("%s",
                # err)``. Mirrors the discipline at the redirect-
                # verify arm below.
                _safe_addr = sanitize_for_log(node.address)
                logger.debug(
                    "find_leader: %s reports no leader known (%d/%d)",
                    _safe_addr,
                    idx + 1,
                    total_nodes,
                )
                return _ProbeMiss(message=f"{_safe_addr}: no leader known", exc=None)
            finally:
                # Safety-net release: if any exception fired before the
                # explicit release above (e.g. _query_leader raised
                # and the catch arms re-raised, or _check_redirect
                # raised ClusterPolicyError), the slot must still
                # come back to the pool.
                if sem_acquired:
                    semaphore.release()

        # Build ``pending`` INSIDE the try frame so a BaseException
        # (synthetic KeyboardInterrupt, outer cancel landing in the
        # bytecode window) raised mid-construction keeps every
        # already-created task tracked and the ``finally:`` cancels +
        # gathers them. Pre-fix the set was built via a comprehension
        # before ``try:`` — a BaseException there orphaned the live
        # tasks (no done-callback observer, unlike ``find_leader``'s
        # ``_observe_drain_exception`` discipline). Mirrors the pool-
        # side hardening that built ``ConnectionPool.initialize``'s
        # ``create_tasks`` inside its own try frame for the same reason.
        pending: set[asyncio.Task[_LeaderHit | _ProbeMiss]] = set()
        winning_address: str | None = None
        policy_error: ClusterPolicyError | None = None
        unexpected_exc: BaseException | None = None
        try:
            # Create all probe tasks up-front so the post-semaphore
            # ``_verify_redirect`` phase can overlap across nodes
            # (the semaphore inside ``_probe_one`` gates only the
            # initial ``_query_leader`` wire dial; verify runs
            # outside the slot). The up-front creation shape is
            # load-bearing for the parallel-sweep optimisation —
            # gating task creation behind the semaphore would
            # serialise verifies one-after-another and defeat the
            # redirect-stampede amortisation documented at
            # ``_probe_one``.
            #
            # The cost is task allocation: for a NodeStore
            # approaching the wire cap of ``_MAX_NODE_COUNT``
            # (10_000), creating all N tasks before the first
            # ``await asyncio.wait`` would burn tens of
            # milliseconds of synchronous loop CPU. Yield every
            # ``_PROBE_TASK_CREATE_YIELD_EVERY`` allocations so a
            # hostile or buggy NodeStore approaching the cap does
            # not monopolise the loop during the allocation burst.
            for idx, n in enumerate(nodes):
                pending.add(asyncio.create_task(_probe_one(idx, n)))
                if (idx + 1) % _PROBE_TASK_CREATE_YIELD_EVERY == 0:
                    await asyncio.sleep(0)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    # ``task.exception()`` returns the exception
                    # without raising. Non-None means the per-probe
                    # coroutine let it propagate — only
                    # ``ClusterPolicyError`` does this intentionally
                    # (deterministic config error → propagate fast).
                    # Any other class is a programming bug; we
                    # re-raise after cancelling siblings.
                    exc = task.exception()
                    if exc is not None:
                        if isinstance(exc, ClusterPolicyError):
                            if policy_error is None:
                                policy_error = exc
                            continue
                        unexpected_exc = exc
                        break

                    outcome = task.result()
                    if isinstance(outcome, _LeaderHit):
                        winning_address = outcome.address
                        break
                    # ``_ProbeMiss`` → accumulate.
                    errors.append(outcome.message)
                    if outcome.exc is not None:
                        per_node_excs.append(outcome.exc)

                if (
                    winning_address is not None
                    or policy_error is not None
                    or unexpected_exc is not None
                ):
                    break
        finally:
            # Cancel siblings on first success, on policy-error, on
            # programming-bug raise, and on outer cancel of this
            # coroutine. ``_query_leader``'s shielded ``finally:``
            # drains each cancelled socket within 100 ms, so
            # cancellation does not leak FDs even under leader-flip
            # stampede. Await siblings so their drain completes
            # within the same ``_find_leader_impl`` scope.
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if unexpected_exc is not None:
            raise unexpected_exc
        if winning_address is not None:
            if policy_error is not None:
                # A sibling probe redirected to a policy-rejected
                # target before the winning probe self-confirmed as
                # leader. The probe-site DEBUG log at
                # ``_check_redirect`` named the rejected address, but
                # without a WARNING here a SIEM watching for
                # security-adjacent signals only sees the DEBUG hit
                # and misses the "the sweep won past it" context.
                # Sanitize the policy_error text via
                # ``sanitize_for_log`` (CWE-117 — a server-supplied
                # address inside ``policy_error.args`` could otherwise
                # split a log record).
                logger.warning(
                    "find_leader: dropped policy rejection during successful "
                    "sweep — rejected redirect=%s, winning leader=%s",
                    sanitize_for_log(str(policy_error)),
                    sanitize_for_log(winning_address),
                )
            # Populate the leader-tracker cache so the next
            # ``find_leader`` takes the fast path (one probe) instead
            # of running the full parallel sweep again. Mirrors
            # ``connector.go:214``'s ``c.lt.SetLeaderAddr(...)``.
            self._set_last_known_leader(winning_address)
            return winning_address
        if policy_error is not None:
            # Invalidate any cached leader before raising: a prior sweep
            # may have cached an address that today only redirects to a
            # policy-rejected target. Without this, the next call hits
            # the fast path against the same cache entry, wastes a
            # round-trip, and re-raises the same ClusterPolicyError.
            # Mirrors the fast-path arm's ``_set_last_known_leader(None)``
            # discipline above.
            self._set_last_known_leader(None)
            # Even on policy rejection, surface the accumulated per-node
            # transport history so an operator can distinguish "policy
            # rejected the leader on an otherwise-healthy cluster" from
            # "policy rejected the leader on a half-down cluster". The
            # chaining discipline mirrors the no-policy aggregate arm
            # below; ``raise ... from ...`` preserves the original
            # exception class so callers branching on
            # ``isinstance(exc, ClusterPolicyError)`` continue to match.
            if len(per_node_excs) > 1:
                raise policy_error from _bounded_group(
                    "find_leader: per-node failures alongside policy rejection",
                    per_node_excs,
                )
            if per_node_excs:
                raise policy_error from per_node_excs[0]
            raise policy_error

        joined = "; ".join(errors)
        if len(joined) > _MAX_AGGREGATE_ERROR_PAYLOAD:
            overflow = len(joined) - _MAX_AGGREGATE_ERROR_PAYLOAD
            joined = (
                joined[:_MAX_AGGREGATE_ERROR_PAYLOAD]
                + f"... [aggregate truncated, {overflow} chars]"
            )
        # Aggregate-failure WARNING. Per-node probes are at DEBUG so
        # healthy sweeps do not spam logs, but the all-nodes-failed
        # outcome is the one event operators paged on cluster-wide
        # unreachable need to see at default verbosity. The errors
        # string is already capped above so the log line is bounded.
        # Sanitise server-supplied text against log-injection: a
        # hostile FailureResponse.message containing ``\n`` would
        # otherwise split the log line in syslog / journald.
        # sanitize_for_log escapes LF / CR; the exception messages
        # below keep the original wording for interactive debugging.
        logger.warning(
            "cluster: leader discovery failed across %d nodes; errors=%s",
            total_nodes,
            sanitize_for_log(joined),
        )
        # Chain via ``BaseExceptionGroup`` when more than one node
        # contributed a real exception (the no-leader-known arm
        # produces no exception, only an entry in ``errors``). Single-
        # exception case keeps the narrow chain so existing callers
        # that branch on ``e.__cause__`` type continue to work.
        # No-exception case (every node returned no-leader-known)
        # raises with no chain — the message itself is the
        # diagnostic. Mirrors ``ConnectionPool.initialize``'s discipline.
        if len(per_node_excs) > 1:
            raise ClusterError(f"Could not find leader. Errors: {joined}") from _bounded_group(
                "find_leader: per-node failures", per_node_excs
            )
        if per_node_excs:
            raise ClusterError(f"Could not find leader. Errors: {joined}") from per_node_excs[0]
        raise ClusterError(f"Could not find leader. Errors: {joined}")

    async def _safe_node_snapshot(self) -> tuple[_StoreNodeInfo, ...]:
        """Defensive snapshot of the node store's current view.

        The :class:`NodeStore` Protocol's ``get_nodes`` contract says
        the returned ``Sequence`` "will not be mutated after return"
        — but that is a constraint on the implementer, not a
        defensive guarantee from the caller. A third-party
        ``NodeStore`` returning its internal storage and mutating it
        from a concurrent ``set_nodes()`` would produce a torn
        iteration deep inside the parallel sweep. ``MemoryNodeStore``
        and ``YamlNodeStore`` already return tuples; the cost of the
        snapshot is O(N) at typical 3-7 node store sizes.

        Use this helper at every call site that iterates the node
        store rather than ``await self._node_store.get_nodes()``
        directly, so future code paths inherit the snapshot
        discipline.

        Bound the store call by ``dial_timeout`` so a misbehaving
        third-party (etcd-/consul-backed) store cannot pin
        ``find_leader`` indefinitely. Symmetric with the
        ``asyncio.timeout`` envelope around every other awaitable in
        ``cluster.py`` (``_query_leader``, ``_verify_redirect``, the
        cached-fast-path probe). In-tree stores return synchronously
        so the wrap is a no-op for them; the bound matters only for
        third-party implementations whose ``get_nodes`` does
        blocking I/O. Mirrors go-dqlite's ``NodeStore.Get(ctx)``
        contract.
        """
        async with asyncio.timeout(self._dial_timeout):
            return tuple(await self._node_store.get_nodes())

    async def _query_leader(
        self, address: str, *, trust_server_heartbeat: bool = False
    ) -> str | None:
        """Query a node for the current leader.

        Raises ``OSError`` (or a subclass) if the dial itself fails;
        ``DqliteConnectionError`` / ``ProtocolError`` /
        ``OperationalError`` for handshake-or-later failures. The
        caller (``_probe_one``) attributes each failure class to the
        per-node aggregate ``ClusterError``. Returns ``None`` when
        the node is reachable but reports no leader (the
        "unknown leader" branch the wire-level ``LeaderResponse``
        handles).

        Pre-fix this swallowed pre-handshake ``OSError`` as ``None``,
        losing per-node attribution: a node that consistently
        refused connection contributed to the aggregate as
        ``"<addr>: no leader known"`` rather than
        ``"<addr>: ConnectionRefusedError"``. Operators reading the
        aggregate cannot remediate without distinguishing
        "node unreachable" from "node up, no leader elected".
        """
        # Let OSError (subsumes TimeoutError, ConnectionRefused,
        # BrokenPipe, etc.) propagate to the caller. ``_probe_one``
        # already wraps this call in a try/except that classifies
        # transport errors per node into the aggregate ClusterError.
        #
        # ``writer = None`` initialised before the try so the finally
        # below always sees a defined name; the dial happens INSIDE
        # the try so a cancel landing between dial-success and
        # protocol-construction lands inside the try-frame and the
        # finally-block drains the writer rather than orphaning it.
        # Use ``asyncio.timeout`` (cancel-scope semantics) rather
        # than ``asyncio.wait_for`` (which discards the inner-task
        # result on outer-cancel — the canonical leak shape for
        # this class of bug). Mirrors the discipline already used
        # by ``_acquire_admin_protocol``.
        writer = None
        try:
            async with asyncio.timeout(self._dial_timeout):
                reader, writer = await open_connection(address, dial_func=self._dial_func)
            protocol = DqliteProtocol(
                reader,
                writer,
                timeout=self._timeout,
                trust_server_heartbeat=trust_server_heartbeat,
                max_total_rows=self._max_total_rows,
                max_continuation_frames=self._max_continuation_frames,
                max_message_size=self._max_message_size,
                address=address,
            )
            # Probe-only handshake: write the protocol version and
            # IMMEDIATELY issue the LeaderRequest, skipping the
            # ``ClientRequest`` registration. The C server's
            # ``handle_leader`` does not require ``g->client_id`` to
            # be set, so probes against non-leader peers no longer
            # allocate a per-client server slot. Saves one wire RTT
            # per probe and removes a stream of bogus
            # ``ClientRequest`` log lines on non-leader nodes.
            # The chosen leader gets the full registration via
            # ``handshake()`` inside ``DqliteConnection.connect``.
            await protocol.negotiate_protocol_only()
            node_id, leader_addr = await protocol.get_leader()

            # Upstream dqlite's ``raft_leader`` normally pairs ``id``
            # and ``address`` (both filled in for a known leader, both
            # zero/NULL for "no leader"). The server substitutes
            # ``""`` for NULL on the wire. The ``(node_id != 0,
            # leader_addr == "")`` shape is reachable on a real
            # follower after a ``RAFT_NOMEM`` from
            # ``recvUpdateLeader`` (raft/recv.c): step 1 sets
            # ``current_leader.id``; step 4 mallocs the address.
            # If the malloc fails, the follower reports ``(id=N,
            # address="")`` until the next AppendEntries arrival
            # retries the update. The Go and C clients both treat
            # this as "leader unknown"; do the same here so the
            # operator does not see a connection-killing
            # ProtocolError on a recoverable cluster window. The
            # wire decoder logs a WARNING for forensic visibility.
            if node_id != 0 and not leader_addr:
                logger.debug(
                    "query_leader: %s reports leader_id=%d with empty "
                    "address (RAFT_NOMEM transient — treating as "
                    "'no leader known')",
                    sanitize_for_log(address),
                    node_id,
                )
                return None
            if node_id == 0 and leader_addr:
                # Mirror arm: the inverse illegal shape. Upstream
                # ``raft_leader`` never writes a non-empty address with
                # id=0, so a peer returning this is either confused or
                # hostile. Reject symmetrically so the redirect target
                # is not trusted without a matching id.
                #
                # Same helper rationale as the arm above:
                # ``sanitize_for_log`` for the DEBUG log record (LF /
                # Tab escaped, CWE-117), ``_sanitize_display_text`` for
                # the exception-message rendering below.
                logger.debug(
                    "query_leader: %s returned malformed redirect (node_id=0, address=%r)",
                    sanitize_for_log(address),
                    sanitize_for_log(leader_addr),
                )
                raise ProtocolError(
                    f"server {_sanitize_display_text(address)} returned "
                    f"address {_sanitize_display_text(leader_addr)!r} with "
                    f"node_id=0; expected both or neither"
                )
            if leader_addr:
                return leader_addr
            # node_id=0 and empty address: no leader known
            return None
        finally:
            # ``writer is None`` when the dial failed before
            # assignment (ConnectionRefused inside the
            # ``async with asyncio.timeout(...)`` block) or when
            # cancel landed before the unpack completed — in either
            # case there is nothing to drain. ``return`` inside a
            # ``finally`` would silently discard a propagating
            # exception from the try-body; gate the drain block on
            # the writer presence instead.
            if writer is not None:
                writer.close()

                # close() is fire-and-forget; without the bounded
                # wait_closed() the transport sits in FIN-WAIT until the
                # OS reclaims it, which adds up under heavy leader-probe
                # churn (pool warm-up after a leader flip). Cap the wait
                # at 100 ms: a responsive peer drains FIN/ACK in microseconds
                # and a slow peer must never hold up leader discovery — the
                # OS will reap the socket later.
                #
                # Shield the drain against an outer cancellation: a
                # ``find_leader`` cancelled by a TaskGroup sibling failure
                # (or any caller-level ``asyncio.timeout``) would otherwise
                # propagate CancelledError out of the ``await`` between
                # ``writer.close()`` (sync) and the awaited
                # ``wait_closed``, leaving the reader task spawned by
                # ``asyncio.open_connection`` orphaned. Under leader
                # stampede scenarios that leaks per-probe. Shielding runs
                # the drain to completion within its 100 ms budget even
                # during shutdown; the outer cancel still propagates past
                # this ``finally`` as expected.
                #
                # Wrap the inner ``wait_for`` in an explicit Task with a
                # done-callback that observes ``.exception()`` so the
                # ``TimeoutError`` (or ``CancelledError`` on shutdown) is
                # never an "unobserved task exception" at GC. Mirrors the
                # ``_clear_slot`` done-callback discipline used by the
                # single-flight find-leader slot map elsewhere in this
                # file. Without the explicit observer, an outer cancel
                # mid-shield orphans the inner Task with a
                # ``TimeoutError`` that asyncio's task-finalisation
                # logger emits as "Task exception was never retrieved".
                #
                # Bounded-tail invariant: when the outer await asyncio.shield
                # is itself cancelled, ``inner_drain`` continues running for
                # up to ``_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS`` (100 ms)
                # before self-terminating via the inner ``asyncio.timeout``
                # cancel scope. Under leader-probe stampede this leaves a
                # 100 ms tail of background work surviving outer-cancel
                # shutdown — bounded by the slot count and the per-drain
                # deadline; intentional to avoid socket leaks under cancel.
                #
                # Use ``asyncio.timeout`` cancel-scope semantics inside
                # the inner Task rather than ``asyncio.wait_for`` so a
                # future refactor that gives ``wait_closed()`` a return
                # value does not silently lose it on outer cancel.
                # Mirrors the discipline at ``protocol.py::_send`` and
                # ``_read_data``.
                async def _drain() -> None:
                    async with asyncio.timeout(_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS):
                        await writer.wait_closed()

                inner_drain: asyncio.Task[None] = asyncio.ensure_future(_drain())
                inner_drain.add_done_callback(_observe_drain_exception)
                with contextlib.suppress(OSError, TimeoutError):
                    await asyncio.shield(inner_drain)

    async def _verify_redirect(
        self,
        hint_address: str,
        *,
        trust_server_heartbeat: bool = False,
        timeout: float | None = None,
    ) -> str | None:
        """Re-probe a redirect target to confirm it self-identifies as
        leader before trusting it.

        Mirrors go-dqlite's ``connector.go::connectAttemptOne``
        redirect re-probe (lines 285-294), trimmed to the LEADER RPC
        only. Returns ``hint_address`` on self-confirmation, ``None``
        on mismatch / unreachable / any transport-level failure (the
        sweep falls through to other nodes; verification failure is
        not fatal).

        Why: ``_query_leader`` returns whatever address a peer claims
        is leader, but a stale-state peer can hand back a node that
        no longer holds leadership. Without re-verification, the
        sweep returns the stale hint, ``connect()`` opens a full
        connection and runs Open, the server responds with a
        leader-change error, and the retry loop kicks in — wasted RTT
        plus a misleading aggregate-error chain pointing at the
        REDIRECTED address. With this re-probe, the stale hint is
        discarded inside the per-node probe and the sweep fans out
        via the parallel-sweep machinery.

        Implementation: delegates the dial+handshake+LEADER RPC to
        ``_query_leader`` so transport discipline (bounded shutdown
        drain wrapped in ``asyncio.shield``) and test-time mocking
        (``patch.object(cluster, "_query_leader", ...)``) apply
        uniformly. The verification adds one decision: the responder
        must report ITSELF as leader, otherwise we discard the hint.

        ``timeout`` selects the per-call wait_for envelope. Defaults
        to ``self._dial_timeout`` for the sweep call site in
        ``_probe_one``.

        Budget composition (post-semaphore-release restructure): the
        ``_probe_one`` semaphore is released after the initial
        ``_query_leader`` returns and BEFORE this verify dials —
        so the verify runs SIBLING, not NESTED, to the initial probe's
        ``wait_for(attempt_timeout)``. The per-probe wall-clock budget
        is therefore ``attempt_timeout + dial_timeout``, not
        ``attempt_timeout`` as the original code structured it.
        Operators sizing ``attempt_timeout`` for a fast control-plane
        should budget this composition explicitly. The verify is
        still bounded by the outer ``find_leader`` ``self._timeout``
        envelope; only the per-probe accounting drifted from the
        original "nested" framing. The cached-leader fast-path call
        site is also sequential to its own initial probe and passes
        ``self._attempt_timeout`` explicitly because that path has
        no semaphore in play.
        """
        effective_timeout = self._dial_timeout if timeout is None else timeout
        try:
            # ``async with asyncio.timeout(...)`` (cancel-scope
            # semantics) rather than ``asyncio.wait_for`` (which would
            # discard the verified-address result on outer-cancel and
            # defeat the very cache the fast-path arm populates).
            # Mirrors the fast-path's own discipline above and the
            # other cancel-scope sibling sites.
            async with asyncio.timeout(effective_timeout):
                reported = await self._query_leader(
                    hint_address, trust_server_heartbeat=trust_server_heartbeat
                )
        except (
            DqliteConnectionError,
            ProtocolError,
            OperationalError,
            OSError,
            TimeoutError,
            ValueError,
        ):
            # ``ValueError`` covers the ``parse_address`` rejection of
            # a malformed redirect hint: the inner ``_query_leader``
            # call dials the hint, ``open_connection`` calls
            # ``parse_address``, and a hostile peer's address that
            # fails the syntactic gate raises ``ValueError`` here. Map
            # to ``None`` so the sweep falls through to the next
            # candidate, matching the rest of the verify-failure
            # outcomes.
            return None
        if reported and _addr_equiv(reported, hint_address):
            return hint_address
        # Stale or pointing elsewhere. Log both addresses through
        # ``sanitize_for_log`` so a hostile peer can't inject CRLF /
        # Tab / control-chars into operator-facing logs (CWE-117).
        # ``_sanitize_display_text`` preserves LF / Tab for
        # exception-message readability — wrong helper for logger
        # records.
        logger.debug(
            "verify_redirect: %s reports leader=%s (stale hint, falling through)",
            sanitize_for_log(hint_address),
            sanitize_for_log(reported) if reported else "<none>",
        )
        return None

    async def connect(
        self,
        database: str = "default",
        *,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        max_attempts: int | None = None,
        max_elapsed_seconds: float | None = None,
        policy: RedirectPolicy | None = None,
        max_message_size: int | None = None,
    ) -> DqliteConnection:
        """Connect to the cluster leader.

        Returns a connection to the current leader. ``max_total_rows``,
        ``max_continuation_frames``, and ``trust_server_heartbeat`` are
        forwarded to the underlying :class:`DqliteConnection` so callers
        (including :class:`ConnectionPool`) can tune security/DoS
        governors from one place.

        ``max_attempts`` overrides the default
        :data:`_DEFAULT_CONNECT_MAX_ATTEMPTS`. ``None`` selects the
        package default of 3 — covers one leader change plus one
        transport hiccup.

        ``max_elapsed_seconds`` is the total wall-clock cap on the
        retry loop. ``None`` (default) means only ``max_attempts``
        governs termination. Set to a positive finite number to
        abort the retry loop once cumulative elapsed time crosses
        the budget. This is the go-dqlite parity knob — go-dqlite
        retries until the caller's context expires; here, callers
        pass the budget explicitly so misconfiguration (no parent
        timeout, ``max_elapsed_seconds=None``) cannot become a
        silent infinite hang. Compose with ``max_attempts`` to
        bound either dimension first.

        Per-iteration backoff is capped at 1 s (matching
        go-dqlite's ``Config.BackoffCap``); other
        ``retry_with_backoff`` users keep the 10 s default.

        Each attempt's failure is logged at DEBUG level with the
        attempted leader address and the error, so operators can
        enable debug logging to diagnose cluster churn instead of
        seeing only the final exception.

        Comparison to go-dqlite
        -----------------------

        Two structural divergences from go-dqlite's
        ``Connector.Connect`` (``/internal/protocol/connector.go``)
        operators porting code should know about:

        1. Cancel-propagation discipline. go-dqlite checks
           ``ctx.Done()`` only at the TOP of each retry iteration
           (``attempt > 1``), so the FIRST attempt always runs even
           if the context was already cancelled when ``Connect``
           was invoked. Python's asyncio cancel propagates through
           every ``await`` checkpoint, so an outer
           ``asyncio.timeout(0)`` wrapping ``cluster.connect()``
           lands the cancel BEFORE the first attempt. The first
           attempt never runs; the caller sees a bare
           ``TimeoutError`` with no per-attempt log breadcrumb.
           This is the CORRECT shape for the Python ecosystem
           (asyncio's structured-concurrency cancel contract is the
           source of truth); callers porting from Go that rely on
           "one attempt is always made" semantics must adapt.

        2. Discovery vs. connection: double-dial cost. go-dqlite's
           ``connectAttemptOne`` adopts the probe socket on success
           (``case address:`` arm at ``connector.go:392-409`` sends
           ``EncodeClient`` on the OPEN socket). Python's design
           returns an address from ``_query_leader`` and re-dials
           in ``_connect_impl`` — every successful
           ``ClusterClient.connect()`` pays one extra TCP three-way
           + handshake RTT to the leader. The single-flight slot
           amortises the probe sweep across N concurrent
           ``connect()`` callers, so the cost grows linearly with
           ``max_size`` for pool warm-up but not per-acquire when
           probes overlap. Operators sizing ``attempt_timeout`` on
           high-RTT WAN deployments should budget for the extra
           round trip.

        Args:
            close_timeout: Budget (seconds) for the transport-drain
                during ``close()``. After ``writer.close()`` the
                local side of the socket is gone; ``wait_closed`` is
                best-effort cleanup. The 0.5s default is sized for
                LAN; increase for WAN deployments where FIN/ACK
                round-trip is slower, or decrease to tighten
                SIGTERM-shutdown budgets. See
                ``DqliteConnection.__init__`` for full rationale.
            policy: Optional per-call ``RedirectPolicy`` override that
                applies to this connect attempt only. Mirrors the
                ``policy=`` kwarg on :meth:`find_leader`,
                :meth:`cluster_info`, and :meth:`leader_info`.
                ``None`` (default) falls back to the instance-level
                ``redirect_policy`` configured at construction. Use
                when an audit-mode caller wants to tighten the
                policy for a single connect without mutating
                instance state. Note: ``find_leader``'s single-flight
                slot key includes the callable identity, so passing a
                per-call ``policy`` defeats the in-flight collapse
                for that call.
        """
        # Reject ``bool`` before the < 1 check so ``True``/``False``
        # don't silently coerce to 1/0. Mirrors the discipline in
        # ``ConnectionPool.__init__``.
        if max_attempts is not None and (
            isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
        ):
            raise TypeError(f"max_attempts must be int or None, got {type(max_attempts).__name__}")
        attempts_cap = max_attempts if max_attempts is not None else _DEFAULT_CONNECT_MAX_ATTEMPTS
        if attempts_cap < 1:
            # Wording mirrors ``ConnectionPool.__init__``'s validator
            # so operator-facing error parsing matches whichever layer
            # validated first.
            raise ValueError(f"max_attempts must be at least 1 if provided, got {attempts_cap}")
        # ``max_elapsed_seconds`` validation is duplicated here (in
        # addition to the validator inside ``retry_with_backoff``) so
        # a misconfiguration surfaces at the public connect() call
        # site rather than after attempt 1 burned its time budget.
        # Wording matches ``retry.py`` exactly so test assertions
        # pinning either layer's message stay green.
        if max_elapsed_seconds is not None:
            if isinstance(max_elapsed_seconds, bool) or not isinstance(
                max_elapsed_seconds, (int, float)
            ):
                raise TypeError(
                    f"max_elapsed_seconds must be a number or None, "
                    f"got {type(max_elapsed_seconds).__name__}"
                )
            if not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0:
                raise ValueError(
                    f"max_elapsed_seconds must be a positive finite number, "
                    f"got {max_elapsed_seconds}"
                )

        # Resolve the effective DoS cap with the "per-call overrides
        # instance default; None falls back" idiom already used for
        # ``policy=`` at cluster.py:744-756. Without this resolution
        # the per-call kwarg's ``None`` default silently dropped the
        # ``ClusterClient(..., max_message_size=N)`` constructor value
        # — ``DqliteProtocol`` then resolved ``None`` to
        # ``DEFAULT_MAX_MESSAGE_SIZE`` (the wire default), silently
        # bypassing the operator's DoS-hardening cap on the connect
        # path. Sibling sites at cluster.py:1428 (``_query_leader``)
        # and cluster.py:2828 (admin connect) already consult
        # ``self._max_message_size``; this restores parity.
        effective_max_message_size = (
            max_message_size if max_message_size is not None else self._max_message_size
        )

        attempt_counter = [0]

        async def try_connect() -> DqliteConnection:
            attempt_counter[0] += 1
            attempt = attempt_counter[0]
            leader: str | None = None
            try:
                leader = await self.find_leader(
                    trust_server_heartbeat=trust_server_heartbeat,
                    policy=policy,
                )
                # Pre-validate the leader address explicitly so the
                # "Server redirected to invalid leader address" message
                # only fires on address-shape failures attributable to
                # the server's LeaderResponse. ``DqliteConnection.__init__``
                # validates many other fields (close_timeout, max_total_rows,
                # max_message_size, ...) and raises bare ``ValueError`` for
                # any of them. Catching all ValueErrors at the construction
                # site misattributed client-knob misconfiguration as a
                # server-redirect failure — operator triage went looking
                # at the server when the fault was in their own connect
                # arguments. Let validator ``ValueError``s from the
                # constructor propagate as-is with their field-specific
                # message so operators see what they actually set wrong.
                try:
                    parse_address(leader)
                except ValueError as e:
                    raise ClusterPolicyError(
                        f"Server redirected to invalid leader address: {e}"
                    ) from e
                conn = DqliteConnection(
                    leader,
                    database=database,
                    timeout=self._timeout,
                    dial_timeout=self._dial_timeout,
                    attempt_timeout=self._attempt_timeout,
                    max_total_rows=max_total_rows,
                    max_continuation_frames=max_continuation_frames,
                    trust_server_heartbeat=trust_server_heartbeat,
                    close_timeout=close_timeout,
                    dial_func=self._dial_func,
                    max_message_size=effective_max_message_size,
                )
                try:
                    await conn.connect()
                except BaseException:
                    # CancelledError, KeyboardInterrupt, SystemExit, or
                    # an unexpected exception during handshake. The conn
                    # was constructed and may have published _protocol
                    # before _run_protocol's CancelledError arm
                    # invalidated and scheduled a _pending_drain task
                    # strong-referenced on the conn. Without an explicit
                    # close here, the conn falls out of scope and the
                    # drain task is GC'd mid-flight, producing a
                    # "Task was destroyed but it is pending" warning
                    # at interpreter shutdown.
                    #
                    # Mirrors ``pool.py``'s shielded-cleanup
                    # discipline:
                    #
                    # * CancelledError absorbed (canonical asyncio
                    #   pattern — asyncio re-delivers at the next
                    #   await; the bare ``raise`` below re-delivers
                    #   the original handshake exception).
                    # * OSError / DqliteConnectionError caught and
                    #   logged (transport-class teardown failures
                    #   on a half-built conn are expected).
                    # * KI / SystemExit / unexpected Exception
                    #   subclasses propagate — a wide
                    #   ``suppress(BaseException)`` here would
                    #   silently swallow signal-class shutdowns and
                    #   programming bugs alike, supplanting the
                    #   original handshake exception.
                    try:
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.shield(conn.close())
                    except (OSError, DqliteConnectionError):
                        logger.debug(
                            "ClusterClient.connect cleanup: conn.close() failed",
                            exc_info=True,
                        )
                    raise
                return conn
            except (
                OSError,
                DqliteConnectionError,
                ClusterError,
                ValueError,
                OperationalError,
            ) as exc:
                # Narrow catch: these are the transport- and cluster-level
                # failures the retry loop re-attempts. Anything wider would
                # silently log-and-re-raise programming bugs (TypeError,
                # AttributeError, …) which are better left un-instrumented
                # so the traceback points at the real source. Same pattern
                # as the _socket_looks_dead / _drain_idle narrowings.
                # OSError subsumes TimeoutError / BrokenPipeError /
                # ConnectionError / ConnectionResetError.
                # ``ValueError`` is a defense-in-depth backstop for any
                # malformed-address rejection that escapes the
                # inner ``_probe_one`` / ``_verify_redirect`` translation
                # — keeping the find-leader sweep's narrow
                # ``except`` from leaking ``ValueError`` past PEP 249
                # wrapping at higher layers.
                # ``OperationalError`` is included for the DEBUG
                # breadcrumb only: ``_connect_impl`` rewraps leader-
                # flip OperationalError to DqliteConnectionError but
                # propagates non-leader-flip codes (e.g. unknown
                # database name from ``open_database``'s FailureResponse
                # arm) unwrapped. Without this arm, those reach the
                # caller with no per-attempt log line documenting the
                # leader address and attempt number. The retry
                # classifier below is unchanged — these codes remain
                # non-retryable; only the breadcrumb is gained.
                # Invalidate the leader cache so the NEXT retry's
                # ``find_leader`` runs a fresh sweep rather than a
                # cached fast-path probe against the leader we just
                # failed to handshake / open_database against.
                # Mirrors Go's ``connector.go::Connect`` which clears
                # the leader tracker on per-attempt failure. Gated on
                # ``leader is not None`` — if ``find_leader`` itself
                # failed, it already cleared the cache internally and
                # a second invalidation would be redundant.
                if leader is not None:
                    self._set_last_known_leader(None)
                logger.debug(
                    "ClusterClient.connect attempt %d/%d failed (leader=%r): %s",
                    attempt,
                    attempts_cap,
                    leader,
                    # Server-controlled exception messages can carry LF
                    # (sanitize_server_text preserves it for exception
                    # readability); strip via sanitize_for_log so a
                    # hostile peer cannot split this DEBUG record into
                    # multiple journald lines (CWE-117). Sibling
                    # WARNING below at attempts-exhausted uses the same
                    # shape.
                    sanitize_for_log(_truncate_error(str(exc))),
                )
                raise

        # Retry only transport-level errors. Leader-change OperationalError
        # codes are reclassified into DqliteConnectionError inside
        # DqliteConnection.connect(), so we no longer need OperationalError
        # in the retry set — that avoids amplifying a schema/SQL error
        # into 5 × N_nodes RTTs before propagating. ClusterPolicyError
        # reflects a deterministic configuration mismatch (redirect
        # blocked) and is excluded: retrying would just reproduce it and
        # multiply the wall-clock cost.
        try:
            return await retry_with_backoff(
                try_connect,
                max_attempts=attempts_cap,
                max_delay=_DEFAULT_CONNECT_MAX_DELAY,
                max_elapsed_seconds=max_elapsed_seconds,
                # OSError subsumes TimeoutError / BrokenPipeError /
                # ConnectionError / ConnectionResetError, so a single
                # OSError entry covers every stdlib transport-error shape.
                retryable_exceptions=(
                    DqliteConnectionError,
                    ClusterError,
                    OSError,
                ),
                excluded_exceptions=(ClusterPolicyError,),
            )
        except (DqliteConnectionError, ClusterError, OSError) as exc:
            # Aggregate-failure WARNING. Per-attempt failures log at
            # DEBUG (so a routine leader flip's per-attempt churn does
            # not spam logs at default verbosity), but the
            # all-attempts-exhausted outcome is the one event paged
            # operators need to see at default verbosity.
            #
            # ``sanitize_for_log`` strips LF / CR from the embedded
            # exception text — a hostile peer's
            # ``FailureResponse.message`` can ride into ``str(exc)``
            # via the ``DqliteConnectionError`` rewrap of
            # ``LEADER_ERROR_CODES``; a bare interpolation would split
            # the WARNING across multiple log lines in syslog /
            # journald. Symmetric with the find-leader aggregate
            # WARNING above.
            logger.warning(
                "cluster: connect exhausted %d attempts; last_error=%s: %s",
                attempts_cap,
                type(exc).__name__,
                sanitize_for_log(_truncate_error(str(exc))),
            )
            raise

    async def cluster_info(self, *, policy: RedirectPolicy | None = None) -> list[_WireNodeInfo]:
        """Return the current cluster's node list (id, address, role).

        Sends ``ClusterRequest(format=1)`` to the current leader and
        returns the V1 server list. Each :class:`NodeInfo` carries
        ``node_id``, ``address``, and ``role``. Mirrors the spec-level
        admin operation ``go-dqlite/client.Cluster``.

        Any node can answer this — Raft replicates the configuration
        — but this method asks the leader so the returned view is the
        freshest one available. A stale follower could otherwise
        report a configuration that is one Raft log entry behind.

        Costs one extra ``get_leader`` round-trip (on the already-open
        admin connection) to re-confirm leadership before reading the
        cluster configuration. This protects callers feeding the
        result into ``node_store.set_nodes(...)`` from a mid-RPC
        leader-flip race where the responder is no longer leader and
        the actual current leader sits outside the policy allowlist;
        mirrors the same arm in :meth:`leader_info`.

        ``policy`` is an optional :data:`RedirectPolicy` callable
        applied to every returned address. Nodes whose address fails
        the policy are excluded from the returned list and a
        ``logger.warning`` is emitted with the rejected address. Use
        when the result is fed into ``node_store.set_nodes(...)``: a
        hostile leader could otherwise smuggle attacker-controlled
        addresses into your membership rotation. Pair with
        :func:`default_safe_redirect_policy` /
        :func:`allowlist_policy`.

        ``None`` (default) falls back to the instance-level
        ``redirect_policy`` configured at construction so the same
        gate that guards ``find_leader`` redirect targets also guards
        ``cluster_info`` returns. Pass ``policy=lambda _: True`` (or
        another always-True callable) to disable filtering at the
        per-call level. Mirrors :meth:`leader_info`'s precedence.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            DqliteConnectionError: when the admin connection cannot be
                established (transport timeout / OS error / handshake
                failure). Rewrapped by :meth:`open_admin_connection`
                from the underlying ``TimeoutError`` / ``OSError``.
                Also reachable from :meth:`find_leader`'s cached
                fast-path arm.
            ClusterError: when no leader is reachable across the
                configured node store (same condition that
                :meth:`find_leader` would surface).
            OperationalError: when the leader rejects the request
                (e.g. mid-shutdown).
            ProtocolError: on a wire-level shape mismatch.
        """
        try:
            leader_addr = await self.find_leader(policy=policy)
            async with self.open_admin_connection(leader_addr) as protocol:
                # Re-confirm leadership BEFORE reading the cluster
                # config: between ``find_leader`` returning and this
                # round-trip landing, leadership can flip — the
                # responder we dialled may now be a follower while a
                # different node (possibly one not in the operator's
                # ``redirect_policy`` allowlist) holds leadership.
                # Without the re-confirm the policy filter on the
                # returned nodes is correct but the RESPONDER the
                # cluster RPC was sent to may itself be policy-
                # rejected, and the returned config may even strip
                # the actual current leader from the view (leading
                # operators piping ``cluster_info`` into
                # ``node_store.set_nodes(...)`` to break future
                # ``find_leader`` calls). Mirrors the leader-flip
                # re-verify arm in ``leader_info``.
                node_id, address = await protocol.get_leader()
                # An empty ``address`` is the wire-layer's "leader not
                # currently known" shape — both (0, "") (canonical "no
                # leader" sentinel) and (N, "") (RAFT_NOMEM transient
                # documented at messages/responses.py:441-470). Treat
                # both uniformly: skip the redirect-chase and read
                # cluster configuration from the current responder.
                # Mirrors leader_info's RAFT_NOMEM handling (which
                # returns None on the same shape) and matches Go's
                # ``len(address) == 0`` check in client/leader.go.
                if not _addr_equiv(address, leader_addr) and address:
                    # Leadership flipped: the responder hands back a
                    # different address. Re-validate against the
                    # redirect policy AND re-probe the hinted target
                    # before reading the cluster configuration from
                    # it. The cost is one extra RTT on the leader-flip
                    # path; the no-flip happy path adds only the
                    # ``get_leader`` round-trip to the same already-
                    # open connection.
                    self._check_redirect(address, policy=policy)
                    verified = await self._verify_redirect(
                        address,
                        trust_server_heartbeat=False,
                    )
                    if verified is None:
                        # Stale hint that did not re-confirm.
                        # Surface as a ClusterError rather than
                        # silently returning the original (now
                        # stale) responder's view.
                        self._set_last_known_leader(None)
                        raise ClusterError(
                            "cluster_info: leadership flipped mid-RPC "
                            "and the responder's hint did not re-confirm"
                        )
                    async with self.open_admin_connection(verified) as p2:
                        nodes = await p2.cluster()
                    # Refresh the last-known-leader cache to the
                    # redirect-verified address. Without this update
                    # the next ``find_leader`` either misses the fast
                    # path (cache is None) or hits the stale responder
                    # entry (one extra wasted probe). Performance
                    # regression specifically on the post-leader-flip
                    # path — the very scenario where the leader cache
                    # matters most. Sibling ``leader_info`` updates
                    # the cache in its redirect-verify arm; mirror it
                    # here.
                    self._set_last_known_leader(verified)
                else:
                    nodes = await protocol.cluster()
        except (OperationalError, DqliteConnectionError, ProtocolError):
            # Failure-path invalidation only: a leader step-down
            # mid-RPC surfaces as one of these exception classes and
            # would otherwise leave the last-known-leader cache
            # pointing at a now-stale peer. On the SUCCESS path the
            # responding leader has provably just answered the RPC
            # — the cache stays warm so the next ``find_leader`` hits
            # the fast path (one RTT) rather than running a full
            # sweep (N RTTs). Mirrors go-dqlite's ``Client.Cluster``
            # which does not touch the leader tracker on success.
            self._set_last_known_leader(None)
            raise
        # Fall back to the instance-level policy when no per-call
        # override; matches the precedence used by ``leader_info``
        # and by ``find_leader``'s redirect arms.
        effective_policy = policy if policy is not None else self._redirect_policy
        if effective_policy is None:
            return nodes
        filtered: list[_WireNodeInfo] = []
        for i, node in enumerate(nodes):
            if effective_policy(node.address):
                filtered.append(node)
            else:
                logger.warning(
                    "cluster_info: dropping node %s (id=%d, role=%s) — address rejected by policy",
                    # ``sanitize_for_log`` escapes LF / Tab in
                    # addition to control / bidi / invisible
                    # chars; WARNING-level records reach SIEM /
                    # journald / syslog shippers, so a server-
                    # controlled address must not split into a
                    # forged second line (CWE-117).
                    sanitize_for_log(node.address),
                    node.node_id,
                    node.role.name,
                )
            # Cooperative loop yield. The wire cap on
            # ``ServersResponse.nodes`` is ``_MAX_NODE_COUNT``
            # (10_000 in ``dqlitewire.messages.responses``). A
            # hostile or buggy leader returning a node-list close
            # to that cap would otherwise pin the loop here for
            # tens of milliseconds — the default policy calls
            # ``parse_address`` + ``ipaddress.ip_address`` per
            # node (tens of microseconds each on commodity
            # hardware). Yield every K so the burst does not
            # monopolise the loop. K reuses the find-leader
            # probe-task yield constant so an operator tuning
            # one knob doesn't end up out of sync with the other.
            if (i + 1) % _PROBE_TASK_CREATE_YIELD_EVERY == 0:
                await asyncio.sleep(0)
        return filtered

    async def transfer_leadership(self, target_node_id: int) -> None:
        """Transfer leadership to ``target_node_id``.

        Sends ``TransferRequest(target_node_id)`` to the current
        leader. Raft will demote the current leader and promote
        ``target_node_id`` (which must be a voter and reachable);
        the call returns once the server has accepted the request.
        Election convergence — the new leader being able to accept
        writes — is observable via a subsequent
        :meth:`find_leader` (or :meth:`cluster_info`) probe.

        Mirrors the spec-level admin operation
        ``go-dqlite/client.Transfer``.

        This is an ops-grade primitive — applications doing SQL work
        should not call it. It is exposed here for cluster-management
        tooling and test infrastructure that needs deterministic
        leadership control.

        Args:
            target_node_id: id of the voter to promote. The caller is
                responsible for picking a valid voter; the server
                rejects invalid targets with
                :class:`OperationalError`.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            ClusterError: when no leader is reachable.
            OperationalError: when the server rejects the transfer
                (e.g. target is not a voter, target unreachable,
                cluster mid-flux).
            ProtocolError: on a wire-level shape mismatch.
        """
        # Validate locally before the round-trip so callers see a
        # ``TypeError`` from the dialect/CLI layer they invoked, not
        # a cryptic wire-decode error from the server.
        if isinstance(target_node_id, bool) or not isinstance(target_node_id, int):
            raise TypeError(f"target_node_id must be int, got {type(target_node_id).__name__}")
        if target_node_id < 1:
            # Node id 0 is the upstream "no node" sentinel
            # (``LeaderResponse.node_id == 0`` means "no leader
            # known"); rejecting it client-side keeps the diagnostic
            # at the call site.
            raise ValueError(f"target_node_id must be >= 1, got {target_node_id}")

        # No pre-RPC cache invalidation: the find_leader fast-path
        # already handles the "cache points at a stepped-down peer"
        # case — the cached probe fails with a leader-flip code and
        # the sweep runs in the next attempt, costing one extra probe
        # RTT rather than a full sweep. Pre-invalidating would force a
        # full N-node sweep on every transfer call, including the
        # warm-cached no-op case (a transfer to the same node, or a
        # transfer that the cluster rejects because the target is not
        # a voter). The post-RPC ``finally:`` below still invalidates
        # for the leader-step-down-during-RPC case. Mirrors
        # go-dqlite's ``Client.Transfer`` which does not pre-
        # invalidate the leader tracker.
        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.transfer(target_node_id)
        finally:
            # The connected leader has stepped down (success path) OR
            # the RPC failed (potentially because leadership flipped
            # mid-RPC). In both cases the cached leader is at least
            # suspect; invalidate so the next ``find_leader`` runs a
            # fresh sweep instead of probing the now-stale cache. Was
            # previously success-path only — leader-flip-induced
            # failure left the cache pointing at the rejecter for one
            # wasted RTT on the next call.
            self._set_last_known_leader(None)

    async def leader_info(self, *, policy: RedirectPolicy | None = None) -> LeaderInfo | None:
        """Return the current leader's ``(node_id, address)``, or
        ``None`` if no leader is known.

        Single round-trip against the first reachable node in the
        configured store — cheaper than :meth:`cluster_info` (which
        also returns roles for every node) when the caller only
        needs the leader's id (e.g. to confirm a
        :meth:`transfer_leadership` landed at the expected target).

        Mirrors go-dqlite's ``Client.Leader``. The wire's
        ``LeaderResponse`` body has no role field, by design — see
        :class:`LeaderInfo` vs. :class:`NodeInfo`.

        ``None`` is returned for the legitimate
        ``(node_id=0, address="")`` "no leader yet" reply that the
        server emits during a re-election.

        ``policy`` is an optional :data:`RedirectPolicy` callable
        applied to the responder's self-reported leader address when
        leadership flipped between :meth:`find_leader` and the
        follow-up ``get_leader`` round-trip. Without this gate, a
        hostile follower reached on the second hop could return an
        attacker-controlled address as "leader" and the caller would
        receive that address verbatim — bypassing every check the
        instance-level ``redirect_policy`` was designed to enforce.
        Falls back to ``self._redirect_policy`` when ``None`` (matches
        the precedence used by :meth:`cluster_info` and by
        :meth:`find_leader`'s redirect arms); pass an always-True
        callable to disable filtering at the per-call level.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            ClusterError: when no node in the store responds.
            ClusterPolicyError: when the responder's reported address
                differs from the address we connected to AND fails
                ``policy`` / ``self._redirect_policy``.
        """
        # Reuse find_leader's sweep semantics — it already handles
        # node-store iteration, redirect validation, and the
        # malformed-redirect arms. Then ask the leader itself for its
        # node id (a single extra round-trip against the leader).
        # Open question: could we have find_leader keep the node_id
        # it discards and surface it here? Yes, but that is a wider
        # refactor of the single-flight slot map; deferred.
        try:
            leader_addr = await self.find_leader(policy=policy)
            async with self.open_admin_connection(leader_addr) as protocol:
                node_id, address = await protocol.get_leader()
                if node_id == 0 and not address:
                    # Mid-election: the leader we just connected to
                    # no longer self-identifies as leader. Surface
                    # ``None`` rather than confabulating an answer.
                    return None
                if node_id != 0 and not address:
                    # RAFT_NOMEM transient on the responder: step 1
                    # of ``recvUpdateLeader`` (raft/recv.c) set
                    # ``current_leader.id``; step 4 failed to malloc
                    # the address buffer. ``handle_leader``
                    # null-coerces address to ``""`` on the wire.
                    # Treat as "no leader known" matching the Go and
                    # C clients and the sibling ``_query_leader``
                    # arm.
                    logger.debug(
                        "leader_info: %s reports leader_id=%d with empty "
                        "address (RAFT_NOMEM transient — treating as "
                        "'no leader known')",
                        sanitize_for_log(leader_addr),
                        node_id,
                    )
                    return None
                if node_id == 0 and address:
                    # The remaining malformed shape: ``(0, nonempty)``.
                    # Upstream ``raft_leader`` never emits id=0 with a
                    # non-empty address — a hostile follower's
                    # ``(0, attacker-addr)`` cannot ride past the
                    # policy gate below as a silent ``None``.
                    raise ProtocolError(
                        f"leader_info: malformed (node_id, address) — "
                        f"got id={node_id!r} addr={_sanitize_display_text(address)!r}; "
                        f"node_id=0 must be paired with an empty address"
                    )
                if not _addr_equiv(address, leader_addr):
                    # Leadership flipped between find_leader and this
                    # follow-up get_leader: the responder hands back
                    # a different address. Re-validate against the
                    # redirect policy (a hostile follower could
                    # otherwise tunnel an attacker-controlled address
                    # through this admin path) AND re-probe the hinted
                    # target to confirm it self-identifies as leader —
                    # mirrors the sweep arms' ``_verify_redirect``
                    # discipline (see ``_probe_one`` and the cached
                    # fast-path arm). Without the re-probe, a stale-
                    # hint follower could feed back a node that no
                    # longer holds leadership and the caller would
                    # use the stale id (e.g. to feed
                    # ``transfer_leadership``).
                    self._check_redirect(address, policy=policy)
                    verified = await self._verify_redirect(
                        address,
                        trust_server_heartbeat=False,
                    )
                    if verified is None:
                        # Stale hint that did not re-confirm. Surface
                        # ``None`` — "no leader known" — rather than
                        # the suspect address.
                        return None
                    # The verified target may report its own
                    # ``node_id`` differently than the original hint
                    # implied; re-fetch from the verified responder
                    # so the returned ``LeaderInfo`` is internally
                    # consistent.
                    async with self.open_admin_connection(verified) as p2:
                        vnode_id, vaddress = await p2.get_leader()
                    if vnode_id == 0 and not vaddress:
                        return None
                    if vnode_id != 0 and not vaddress:
                        # RAFT_NOMEM transient on the verified hint —
                        # same shape as the parent arm; treat as "no
                        # leader known".
                        logger.debug(
                            "leader_info: verified hint %s reports "
                            "leader_id=%d with empty address (RAFT_NOMEM "
                            "transient — treating as 'no leader known')",
                            sanitize_for_log(verified),
                            vnode_id,
                        )
                        return None
                    if vnode_id == 0 and vaddress:
                        raise ProtocolError(
                            f"leader_info: malformed (node_id, address) on "
                            f"verified hint — got id={vnode_id!r} "
                            f"addr={_sanitize_display_text(vaddress)!r}; "
                            f"node_id=0 must be paired with an empty address"
                        )
                    # Filter the third-hop address through the same
                    # redirect policy as the first-hop hint. The verified
                    # responder is allowlisted by the operator but its
                    # reported leader can be any peer (cluster mid-flip,
                    # or — worst case — a compromised follower returning
                    # an attacker-controlled address as its own leader
                    # hint). The defence-in-depth contract documented at
                    # the top of this branch ("hostile follower tunneling
                    # an attacker-controlled address") must apply to the
                    # third hop too, mirroring the per-node policy filter
                    # ``cluster_info`` applies at lines 2099-2114.
                    self._check_redirect(vaddress, policy=policy)
                    return LeaderInfo(node_id=vnode_id, address=vaddress)
                return LeaderInfo(node_id=node_id, address=address)
        except (OperationalError, DqliteConnectionError, ProtocolError):
            # Failure-path invalidation only: a leader step-down
            # mid-RPC surfaces as one of these exception classes and
            # would otherwise leave the cache pointing at a now-stale
            # peer. On SUCCESS the responding leader has provably
            # just answered the RPC — the cache stays warm so the
            # next ``find_leader`` hits the fast path (one RTT)
            # rather than running a full sweep. Mirrors go-dqlite's
            # ``Client.Leader`` which does not touch the leader
            # tracker on success.
            self._set_last_known_leader(None)
            raise

    async def add_node(
        self,
        node_id: int,
        address: str,
        *,
        role: NodeRole = NodeRole.SPARE,
    ) -> None:
        """Add a node to the cluster (Raft membership change).

        Mirrors go-dqlite's ``Client.Add`` two-phase shape: the
        underlying ``ADD`` wire op always lands the node as
        ``NodeRole.SPARE`` (a newly-added voter that has not caught
        up with the leader's log cannot vote). If the caller asked
        for a non-spare role, this method follows up with an
        ``ASSIGN`` to promote — same as
        ``go-dqlite/client.go::Client.Add``.

        Both calls go to the leader; the leader-discovery happens
        once per call.

        **Partial-failure recovery.** The two-phase shape is not
        atomicised: if ``ADD`` succeeds but the follow-up ``ASSIGN``
        raises (transient transport, leader flip, target unreachable),
        the cluster's Raft log records the new node as ``SPARE`` and
        the exception propagates. From the caller's perspective, an
        ``add_node(..., role=NodeRole.VOTER)`` raise is
        indistinguishable in class from "ADD never landed", so a
        naive retry loop would trip the upstream "node already exists"
        on the next ADD attempt.

        Recovery: catch the exception, then call
        ``assign_role(node_id, role)`` to converge the partially-
        added node to the requested role. ``assign_role`` is
        idempotent at the upstream level — re-running it against a
        node already at the target role is a no-op. The same shape
        applies to go-dqlite's ``Client.Add`` (the API contract is
        identical).

        Args:
            node_id: Raft id of the new node. Must be >= 1
                (0 is the upstream "no node" sentinel).
            address: TCP ``host:port`` the new node listens on for
                Raft traffic.
            role: target role; defaults to
                :class:`NodeRole.SPARE` (no follow-up assign needed).

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            TypeError / ValueError: on invalid arguments
                (validated client-side before the round-trip).
            ClusterError: when no leader is reachable.
            OperationalError: when the server rejects (e.g. id
                already in cluster, address unreachable). May surface
                from either the ADD phase or the follow-up ASSIGN —
                see Partial-failure recovery above.
            ProtocolError: on a wire-level shape mismatch.
        """
        _validate_node_id(node_id)
        if not isinstance(address, str) or not address:
            raise TypeError(f"address must be a non-empty str, got {type(address).__name__}")
        if not isinstance(role, NodeRole):
            raise TypeError(f"role must be a NodeRole, got {type(role).__name__}")
        # Defer to the in-tree strict address parser for shape
        # validation. Without this, a malformed address (non-numeric
        # port, unbracketed IPv6, whitespace/CRLF, credentials-smuggle
        # ``user@host``, etc.) would reach the server, get stored in
        # the Raft log, and only surface as a connection failure
        # later when some node tries to dial it. Catching at the call
        # site moves the diagnostic from "node X cannot reach node Y
        # three minutes later" to a clean ValueError.
        try:
            parse_address(address)
        except ValueError as exc:
            raise ValueError(f"add_node: invalid address {address!r}: {exc}") from exc

        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.add(node_id, address)
                if role != NodeRole.SPARE:
                    # Mirror go-dqlite's `Client.Add` second phase: ADD
                    # is always implicitly Spare server-side; promote
                    # with a follow-up Assign. Reuses the same admin
                    # connection so the second call lands on the same
                    # leader (avoids a re-election window between the
                    # two requests).
                    await protocol.assign(node_id, role)
        finally:
            # Raft membership changes can interleave with elections,
            # AND a leader-flip-induced failure here leaves the cache
            # pointing at the rejecter. Invalidate on both success and
            # failure paths.
            self._set_last_known_leader(None)

    async def assign_role(self, node_id: int, role: NodeRole) -> None:
        """Change a node's role (promote or demote).

        Mirrors go-dqlite's ``Client.Assign``. Used to promote a
        spare that has caught up, or to demote a voter to standby
        ahead of a controlled :meth:`remove_node`.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            TypeError / ValueError: on invalid arguments.
            ClusterError: when no leader is reachable.
            OperationalError: when the server rejects (e.g. id not
                in cluster, role unchanged, cluster mid-flux).
            ProtocolError: on a wire-level shape mismatch.
        """
        _validate_node_id(node_id)
        if not isinstance(role, NodeRole):
            raise TypeError(f"role must be a NodeRole, got {type(role).__name__}")

        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.assign(node_id, role)
        finally:
            # Promoting STANDBY → VOTER widens the voting set (election
            # window). Failure-path also invalidates so a leader-flip-
            # induced rejection doesn't leave the cache stale.
            self._set_last_known_leader(None)

    async def remove_node(self, node_id: int) -> None:
        """Remove a node from the cluster (Raft membership change).

        Mirrors go-dqlite's ``Client.Remove``. The other half of the
        membership-change surface alongside :meth:`add_node`.
        Removing the current leader requires a prior
        :meth:`transfer_leadership` to a different voter — the
        server otherwise rejects with a not-leader-style error.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            TypeError / ValueError: on invalid arguments.
            ClusterError: when no leader is reachable.
            OperationalError: when the server rejects.
            ProtocolError: on a wire-level shape mismatch.
        """
        _validate_node_id(node_id)

        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.remove(node_id)
        finally:
            # The removed node may have been the cached leader; the
            # server normally rejects removing the connected leader,
            # but defense against future protocol changes that allow
            # it. Unconditional invalidation on both success and
            # failure is the conservative choice.
            self._set_last_known_leader(None)

    async def describe(self, *, address: str | None = None) -> NodeMetadata:
        """Read a node's failure-domain + weight metadata.

        Mirrors go-dqlite's ``Client.Describe``. The describe
        operation is **per-node** (not per-cluster) — it returns the
        connected peer's own metadata. Pass an explicit ``address``
        to describe a specific node; ``None`` describes the current
        leader (matches go-dqlite's typical
        leader-connected-Client pattern).

        Args:
            address: TCP ``host:port`` of the node to describe.
                ``None`` means "describe the leader".

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard before any transport
                activity; reconstruct the client in the target
                process to recover.
            DqliteConnectionError: when the admin connection cannot be
                established. Rewrapped by :meth:`open_admin_connection`
                from the underlying ``TimeoutError`` / ``OSError``.
                Also reachable from :meth:`find_leader`'s cached
                fast-path arm when ``address`` is ``None``.
            ClusterError: when no leader is reachable (only fires
                when ``address=None``).
            OperationalError: when the node rejects the request.
            ProtocolError: on a wire-level shape mismatch.
        """
        # ``describe(address=<specific>)`` bypasses ``find_leader``
        # and targets the named peer — invalidating the leader cache
        # on that path would be over-aggressive (the cache is
        # unrelated to the call). Only the ``address is None`` arm
        # routes through the leader, so only that arm participates
        # in the membership-RPC invalidation discipline.
        #
        # The fork-after-init guard fires here so the per-address
        # path carries the same discipline as the leader-routed
        # sibling (``find_leader`` already guards). Bypassing this
        # gate would leave the instance "alive enough" to describe
        # a specific node but "dead" for leader-routed calls.
        self._check_pid()
        if address is not None:
            # Defer to the in-tree strict address parser for shape
            # validation, mirroring :meth:`add_node`. Without this,
            # a typoed / malformed address (stray whitespace, missing
            # port, unbracketed IPv6, ``user@host`` shape) reaches
            # ``open_admin_connection`` and surfaces deep in the dial
            # path as a ``DqliteConnectionError`` — at a different
            # site than ``add_node``'s clean ``ValueError``. Catching
            # at the call site keeps the three per-address admin
            # methods' operator-facing diagnostics symmetric.
            try:
                parse_address(address)
            except ValueError as exc:
                raise ValueError(f"describe: invalid address {address!r}: {exc}") from exc
        leader_targeted = address is None
        target = address if address is not None else await self.find_leader()
        try:
            async with self.open_admin_connection(target) as protocol:
                response = await protocol.describe()
                return NodeMetadata(
                    failure_domain=response.failure_domain,
                    weight=response.weight,
                )
        except (OperationalError, DqliteConnectionError, ProtocolError):
            if leader_targeted:
                # Failure-path invalidation only: a leader step-down
                # mid-RPC surfaces as one of these exception classes.
                # On SUCCESS the responding leader has provably just
                # answered the RPC — the cache stays warm. Mirrors
                # go-dqlite's ``Client.Describe`` which does not
                # touch the leader tracker on success.
                self._set_last_known_leader(None)
            raise

    async def set_weight(self, weight: int, *, address: str | None = None) -> None:
        """Set a node's weight (leader-election preference).

        Mirrors go-dqlite's ``Client.Weight``. Like :meth:`describe`,
        this operation is **per-node** — pass an explicit ``address``
        to target a specific node, or ``None`` to target the leader.

        Args:
            weight: non-negative integer (uint64 wire-side).
            address: TCP ``host:port`` of the node to update;
                ``None`` targets the leader.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard before any transport
                activity; reconstruct the client in the target
                process to recover.
            TypeError / ValueError: on invalid arguments.
            DqliteConnectionError: when the admin connection cannot be
                established. Rewrapped by :meth:`open_admin_connection`
                from the underlying ``TimeoutError`` / ``OSError``.
                Also reachable from :meth:`find_leader`'s cached
                fast-path arm when ``address`` is ``None``.
            ClusterError: when ``address=None`` and no leader is reachable.
            OperationalError: when the node rejects.
            ProtocolError: on a wire-level shape mismatch.
        """
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise TypeError(f"weight must be int, got {type(weight).__name__}")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")

        # Same per-address fork guard as :meth:`describe`. The
        # leader-routed arm picks up ``find_leader``'s pid check,
        # but the explicit-address arm would otherwise silently
        # succeed in a forked child.
        self._check_pid()
        if address is not None:
            # Defer to the in-tree strict address parser for shape
            # validation, mirroring :meth:`add_node` and the sibling
            # arm of :meth:`describe`. Catches operator-error-class
            # input (stray whitespace, missing port, unbracketed IPv6,
            # ``user@host`` smuggle) at the call site instead of deep
            # in ``open_admin_connection``'s dial path.
            try:
                parse_address(address)
            except ValueError as exc:
                raise ValueError(f"set_weight: invalid address {address!r}: {exc}") from exc
        leader_targeted = address is None
        target = address if address is not None else await self.find_leader()
        try:
            async with self.open_admin_connection(target) as protocol:
                await protocol.weight(weight)
        except (OperationalError, DqliteConnectionError, ProtocolError):
            if leader_targeted:
                # Failure-path invalidation only: a leader step-down
                # mid-RPC surfaces as one of these exception classes.
                # On SUCCESS the responding leader has provably just
                # answered the RPC — the cache stays warm. Per-node
                # form (``address != None``) dials the target
                # directly and never touches the leader cache.
                # Mirrors go-dqlite's ``Client.Weight`` which does
                # not touch the leader tracker on success.
                self._set_last_known_leader(None)
            raise

    async def dump(self, database: str) -> dict[str, bytes]:
        """Dump a database to ``{filename: bytes}``.

        Mirrors go-dqlite's ``Client.Dump`` (``client.go:131``) where
        the database name is a required positional argument — the
        Go API does NOT default it. Defaulting was a footgun: an
        operator running ``await client.dump()`` to back up
        ``analytics`` would silently dump ``default`` instead. Backups
        are critical operations and a forced-explicit signature is
        the right discipline.

        The dump request is sent to the leader by this method as a
        Python design choice — the upstream gateway
        (``handle_dump`` in ``gateway.c``) does NOT call
        ``CHECK_LEADER``, so any cluster member (voter, standby, or
        spare) can serve the request. Operators wanting to back up
        from a spare to avoid leader-CPU contention during the dump
        can drive the wire layer directly via
        :meth:`open_admin_connection` against the target peer. The
        response materialises every file in the database (typically
        two: the database itself and its ``-wal`` sidecar).

        The wire layer enforces caps on file count + per-file size
        + 8-byte content alignment so a hostile peer cannot exhaust
        client memory; multi-GB databases will hit those caps and
        fail with :class:`ProtocolError` at decode. Operators
        should plan a cluster-side snapshot or out-of-band backup
        for very large databases.

        Args:
            database: dqlite database name. Required — pass
                ``"default"`` explicitly if that is the database to
                dump.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard inside :meth:`find_leader`
                before any transport activity; reconstruct the client
                in the target process to recover.
            TypeError: on invalid arguments.
            DqliteConnectionError: when the admin connection cannot be
                established. Rewrapped by :meth:`open_admin_connection`
                from the underlying ``TimeoutError`` / ``OSError``.
                Also reachable from :meth:`find_leader`'s cached
                fast-path arm.
            ClusterError: when no leader is reachable.
            OperationalError: when the server rejects (e.g. unknown
                database name).
            ProtocolError: on a wire-level shape mismatch.
        """
        if not isinstance(database, str) or not database:
            raise TypeError(f"database must be a non-empty str, got {type(database).__name__}")

        try:
            leader_addr = await self.find_leader()
            async with self.open_admin_connection(leader_addr) as protocol:
                return await protocol.dump(database)
        except (OperationalError, DqliteConnectionError, ProtocolError):
            # Failure-path invalidation only: ``dump`` reads a
            # long-lived socket, and a leader step-down mid-dump is
            # plausible. The exception classes catch this case and
            # invalidate so the next ``find_leader`` re-sweeps. On
            # SUCCESS the responding leader has provably just
            # answered the dump RPC — the cache stays warm. Mirrors
            # go-dqlite's ``Client.Dump`` which does not touch the
            # leader tracker on success.
            self._set_last_known_leader(None)
            raise

    @contextlib.asynccontextmanager
    async def open_admin_connection(self, address: str) -> AsyncIterator[DqliteProtocol]:
        """Open a one-shot admin connection to ``address``, yield a
        handshaken :class:`DqliteProtocol`, and tear the socket down on
        exit.

        Public direct-to-node primitive. Mirrors go-dqlite's
        ``NewDirectConnector(id, address, options...).Connect(ctx)``
        (``client.go:358-367``): a freshly handshaken connection to a
        named node, bypassing leader discovery. Used by every admin
        method on this class (``cluster_info``, ``transfer_leadership``,
        ``leader_info``, ``add_node``, ``assign_role``, ``remove_node``,
        ``describe``, ``set_weight``, ``dump``) and available to
        external callers building bespoke admin tooling against a
        specific node.

        Each call opens a fresh socket — no pool, no reuse — so admin
        traffic does not mix with SQL-path connection lifecycle.
        Mirrors :meth:`_query_leader`'s transport discipline (bounded
        shutdown drain wrapped in ``asyncio.shield``) so a cancelled
        outer task does not leak a half-closed socket. ``dial_timeout``
        bounds the TCP-establish phase; ``timeout`` is the per-RPC
        budget on the yielded protocol.

        Args:
            address: ``host:port`` string of the target node.

        Yields:
            A handshaken :class:`DqliteProtocol` ready for admin RPCs.

        Raises:
            InterfaceError: when ``ClusterClient`` is used after fork
                (created in a different process). Raised by the
                ``_check_pid`` fork guard before any transport
                activity; reconstruct the client in the target
                process to recover.
            DqliteConnectionError: when the dial or handshake fails
                (rewrapped from the underlying ``TimeoutError`` /
                ``OSError``).
            OperationalError: when the node rejects the handshake.
            ProtocolError: on a wire-level shape mismatch.
        """
        # Wrap the entire dial+handshake in ``attempt_timeout`` to
        # mirror the SQL path (``connection.py:_connect_impl``) and
        # the Go canonical ``NewDirectConnector.Connect``
        # (``connector.go:148-150``). Without this outer envelope, a
        # slow-handshaking peer (TLS-terminating proxy with delayed
        # welcome, mid-restart node) hangs admin RPCs for up to the
        # per-RPC ``timeout`` (default 10 s) even when the operator
        # sized ``attempt_timeout`` at e.g. 0.5 s for fast control-
        # plane fail-over. The inner ``asyncio.timeout(self._dial_timeout)``
        # still bounds the TCP-establish phase; the protocol's
        # ``self._timeout`` still bounds per-RPC reads on the yielded
        # protocol.
        #
        # Fork-after-init guard: the operator may have configured a
        # ``dial_func`` that captures parent-loop-bound state
        # (e.g. an ``aiohttp.ClientSession``). Reject post-fork
        # BEFORE invoking the operator's callable so the diagnostic
        # site is the dqlite call, not the operator's library.
        self._check_pid()
        protocol: DqliteProtocol | None = None
        writer = None
        try:
            try:
                async with asyncio.timeout(self._attempt_timeout):
                    # Use ``asyncio.timeout`` (cancel-scope semantics)
                    # rather than ``asyncio.wait_for`` (which discards
                    # the inner-task result on outer-cancel — orphans
                    # ``(reader, writer)`` if the outer cancel lands
                    # between dial-resolve and unpack). Mirrors the
                    # ``_query_leader`` discipline.
                    try:
                        async with asyncio.timeout(self._dial_timeout):
                            reader, writer = await open_connection(
                                address, dial_func=self._dial_func
                            )
                    except TimeoutError as e:
                        # Dial-specific timeout (the inner
                        # ``asyncio.timeout(self._dial_timeout)`` fired).
                        # Re-raised inside the outer envelope so the
                        # outer attempt_timeout arm below can
                        # distinguish handshake-stall from dial-stall.
                        raise DqliteConnectionError(
                            f"Connection to {_sanitize_display_text(address)} timed out"
                        ) from e
                    protocol = DqliteProtocol(
                        reader,
                        writer,
                        timeout=self._timeout,
                        trust_server_heartbeat=self._trust_server_heartbeat,
                        max_total_rows=self._max_total_rows,
                        max_continuation_frames=self._max_continuation_frames,
                        max_message_size=self._max_message_size,
                        address=address,
                    )
                    # NOTE: unlike ``DqliteConnection._connect_impl``,
                    # this method does NOT null ``writer`` after the
                    # ``DqliteProtocol(...)`` hand-off. The protocol is
                    # ``yield``-ed (not stored on ``self``) so there is
                    # no later ``conn.close()`` / ``_abort_protocol``
                    # path that would walk the protocol's writer
                    # reference and drain the transport. The outer
                    # ``finally`` is the only place the writer can be
                    # closed; nulling here would orphan the writer on
                    # the success path.
                    #
                    # Speak only the 8-byte version on direct-admin
                    # paths. Mirrors go-dqlite's ``client.New`` /
                    # ``NewDirectConnector.Connect``
                    # (``client/client.go:56-75``,
                    # ``internal/protocol/connector.go:316-337``)
                    # which send the version write and stop there.
                    # ``ClientRequest`` registration costs an extra
                    # RTT and allocates a server-side
                    # ``g->client_id`` slot retained until TCP close;
                    # no admin RPC on this protocol (``add``,
                    # ``remove``, ``assign``, ``transfer``,
                    # ``describe``, ``weight``, ``dump``, ``cluster``,
                    # ``get_leader``) reads ``self._client_id``, so
                    # the registration is pure waste under monitoring
                    # stampedes. Sibling ``_query_leader`` already
                    # uses the same lighter discipline.
                    await protocol.negotiate_protocol_only()
            except TimeoutError as e:
                # Outer ``attempt_timeout`` fired AFTER the dial
                # completed — i.e. the handshake stalled. Emit a
                # distinct message so an operator paging on
                # ``Connection ... timed out`` does not conflate dial-
                # vs handshake-stall diagnostics.
                raise DqliteConnectionError(
                    f"Admin handshake to {_sanitize_display_text(address)} exceeded attempt_timeout"
                ) from e
            except OSError as e:
                raise DqliteConnectionError(
                    f"Failed to connect to {_sanitize_display_text(address)}: {e}"
                ) from e
            yield protocol
        finally:
            if writer is not None:
                writer.close()

                # Use ``asyncio.timeout`` cancel-scope semantics inside
                # the inner Task rather than ``asyncio.wait_for`` so a
                # future refactor that gives ``wait_closed()`` a return
                # value does not silently lose it on outer cancel.
                # Mirrors the discipline at ``protocol.py::_send`` and
                # ``_read_data``.
                async def _drain() -> None:
                    async with asyncio.timeout(_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS):
                        await writer.wait_closed()

                inner_drain: asyncio.Task[None] = asyncio.ensure_future(_drain())
                inner_drain.add_done_callback(_observe_drain_exception)
                with contextlib.suppress(OSError, TimeoutError):
                    await asyncio.shield(inner_drain)


def allowlist_policy(addresses: Iterable[str]) -> RedirectPolicy:
    """Build a redirect policy that accepts only the given addresses.

    Useful for the common case: "only allow redirects to hosts I've
    explicitly seed-listed." Addresses are normalized via
    :func:`parse_address` and compared as ``(host, port)`` tuples, so
    bracketed and unbracketed IPv6 forms (``[::1]:9001`` vs
    ``::1:9001``) match each other and hostname casing is irrelevant
    (the parser lower-cases hosts). Callers that need CIDR / DNS /
    wildcard matching should supply their own callable.

    Each entry is parsed at construction time; a malformed entry
    raises :class:`ValueError` from ``allowlist_policy`` itself, so
    typos surface at the operator's config-load site rather than as
    a silent rejection of a legitimate redirect later. A malformed
    *runtime* redirect target returns ``False`` (the policy is a
    safety filter; we do not let a hostile server crash it by
    sending garbage).

    Accepts any iterable (list, set, tuple, generator, dict_keys).
    The iterable is materialized into a frozen set once, so passing
    a generator is safe — the returned closure doesn't re-iterate.
    """
    parsed: list[tuple[str, int]] = []
    for raw in addresses:
        try:
            parsed.append(parse_address(raw))
        except ValueError as e:
            raise ValueError(f"allowlist_policy: invalid address {raw!r} ({e})") from e
    allowed = frozenset(parsed)

    def policy(addr: str) -> bool:
        try:
            return parse_address(addr) in allowed
        except ValueError:
            return False

    return policy


def default_safe_redirect_policy(
    *,
    include_rfc1918: bool = True,
    include_loopback: bool = False,
) -> RedirectPolicy:
    """Build the recommended-default redirect policy.

    Rejects link-local (e.g. AWS / GCP metadata IP
    ``169.254.169.254``) by default — the canonical SSRF target.
    By default also rejects loopback (a redirect to ``127.0.0.1`` is
    rarely a legitimate cluster topology), but operators running
    integration tests that bind to loopback can pass
    ``include_loopback=True``.

    ``include_rfc1918`` defaults to ``True`` because most production
    dqlite clusters live on RFC 1918 / private subnets and would
    otherwise be locked out. Operators with internet-facing clusters
    can pass ``include_rfc1918=False`` to harden further.

    Use as the default value of ``redirect_policy`` when no
    operator-specific allowlist is configured. Pair with
    :func:`allowlist_policy` for stricter allowlists.

    Returns a callable ``RedirectPolicy``: ``True`` if the address
    is allowed, ``False`` to reject. Malformed input returns
    ``False`` (a hostile server cannot crash the policy by sending
    garbage).

    SSRF caveats
    ------------
    The policy is applied to the literal address string BEFORE name
    resolution and BEFORE IPv6 tunnel unwrapping. Two specific
    bypasses to be aware of:

    1. **DNS hostnames pass through.** Any address whose host is a
       hostname rather than an IP literal is accepted (the policy
       cannot judge what the hostname will resolve to without an
       additional resolver round-trip). A hostname that resolves
       at dial time to ``169.254.169.254`` (or any other rejected
       IP) is therefore **NOT** blocked. Deployments requiring
       strict SSRF defense should pair this policy with a custom
       resolver / split-horizon DNS / a hostname-rejecting policy
       (``allowlist_policy`` is the simplest path).
    2. **IPv6 tunnel encapsulations.** ``::ffff:<ipv4>``, 6to4
       (``2002::/16``), and Teredo (``2001::/32``) wrappings of
       SSRF-class IPv4 targets are all unwrapped before classifying,
       so a metadata IP smuggled inside any of these envelopes is
       blocked. Other IPv6 tunnel modes (ISATAP, NAT64 ``64:ff9b::/96``)
       are not unwrapped; operators on networks using those should
       prefer :func:`allowlist_policy`.
    """
    import ipaddress

    def policy(addr: str) -> bool:
        try:
            host, _port = parse_address(addr)
        except ValueError:
            return False
        # Try to interpret host as a literal IP. DNS hostnames pass
        # through (the policy is applied BEFORE resolution; the
        # operator's DNS layer is the source of truth for hostname
        # mapping). For literal IPs, apply the address-class filter.
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True  # hostname; rely on DNS / allowlist for finer control
        # Unwrap IPv6 encapsulations of IPv4 addresses so a 6to4 /
        # Teredo / IPv4-mapped wrapping of a metadata IP classifies
        # as the embedded v4 rather than as the wrapper (which is
        # neither link-local nor private by itself).
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            elif ip.sixtofour is not None:
                # ``2002::/16`` per RFC 3056.
                ip = ip.sixtofour
            elif ip.teredo is not None:
                # ``2001::/32`` per RFC 4380. ``ip.teredo`` returns
                # ``(server_v4, client_v4)``; the client field is the
                # tunneled-to address we want to classify.
                _server_v4, client_v4 = ip.teredo
                ip = client_v4
        # Link-local: reject unconditionally (no operator override)
        # because the only realistic use case is an exfiltration
        # target like the metadata endpoint.
        if ip.is_link_local:
            return False
        if ip.is_loopback and not include_loopback:
            return False
        # ``is_private`` covers RFC 1918 plus other private
        # ranges (RFC 4193 ULA, etc.) — same posture for all.
        return not (ip.is_private and not include_rfc1918)

    return policy
