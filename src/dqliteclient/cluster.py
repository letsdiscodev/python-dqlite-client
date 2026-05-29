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

# Aliased to avoid shadowing the user-facing ``node_store.NodeInfo``: same
# field shape but distinct type, used internally to decode ``ServersResponse``.
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

# Returns True to accept a redirect target, False to reject. Invoked
# synchronously inside the probe loop holding the semaphore, so it MUST be
# cheap and non-blocking (no socket I/O, DNS, or directory lookup).
type RedirectPolicy = Callable[[str], bool]

# Three attempts cover one leader change plus one transport hiccup; higher
# counts risk hiding genuine cluster instability.
_DEFAULT_CONNECT_MAX_ATTEMPTS: Final[int] = 3

# Per-iteration backoff cap for connect()'s retry loop. Matches go-dqlite's
# ``Config.BackoffCap`` (1 s); lower than retry.py's 10 s default because
# leader discovery has a tighter latency SLO.
_DEFAULT_CONNECT_MAX_DELAY: Final[float] = 1.0

# Cap on simultaneous in-flight leader probes per sweep so a 500-node cluster
# does not open 500 sockets at once. Mirrors go-dqlite's ConcurrentLeaderConns.
_DEFAULT_CONCURRENT_LEADER_CONNS: Final[int] = 10

# Cap per-node error messages before concatenation so a peer returning a
# multi-MB FailureResponse cannot produce an O(N*M) string in every traceback.
_MAX_ERROR_MESSAGE_SNIPPET: Final[int] = 200

# Cap the aggregate per-node error payload: the per-node cap bounds the M
# axis, but node-store size (N) is operator-controlled and unbounded.
_MAX_AGGREGATE_ERROR_PAYLOAD: Final[int] = 16 * 1024

# Cap children in a per-node-failure BaseExceptionGroup so a large cluster
# cannot produce a multi-MB exception graph that survives cross-process pickling.
_MAX_AGGREGATE_CHILDREN: Final[int] = 20

# Yield a scheduler tick every N probe-task creations so a near-cap (10_000)
# NodeStore does not monopolise the loop during the allocation burst.
_PROBE_TASK_CREATE_YIELD_EVERY: Final[int] = 256


def _bounded_group(message: str, excs: list[BaseException]) -> BaseExceptionGroup[BaseException]:
    """Build a BaseExceptionGroup capped at _MAX_AGGREGATE_CHILDREN; excess is
    summarised by a synthetic DqliteError so the chain stays picklable."""
    # Local import to avoid the module-level cluster -> exceptions cycle.
    from dqliteclient.exceptions import DqliteError

    if len(excs) <= _MAX_AGGREGATE_CHILDREN:
        return BaseExceptionGroup(message, excs)
    kept = excs[:_MAX_AGGREGATE_CHILDREN]
    overflow = DqliteError(
        f"... and {len(excs) - _MAX_AGGREGATE_CHILDREN} more per-node failures "
        "(aggregate truncated to keep the exception graph picklable)"
    )
    return BaseExceptionGroup(message, [*kept, overflow])


# SystemRandom ignores random.seed(), so the per-sweep shuffle's stampede
# avoidance survives a downstream seed of the global PRNG.
_cluster_random: Final[random.Random] = random.SystemRandom()

# Budget for the bounded writer-drain in _query_leader; a slow peer must not
# hold up leader discovery.
_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS: Final[float] = 0.1


def _addr_equiv(a: str, b: str) -> bool:
    """Compare host:port via the canonical ``(host, port)`` tuple, falling
    back to literal equality for inputs parse_address rejects."""
    try:
        return parse_address(a) == parse_address(b)
    except ValueError:
        return a == b


def _truncate_error(message: str) -> str:
    # Suffix reports the overflow count (len - max), not total length, to
    # match the wire-layer cap_raw_message SSOT. Sanitise (display variant:
    # keeps LF/Tab, strips control/bidi) so server text cannot smuggle
    # log-splitting chars into raw_message (CWE-117).
    safe = _sanitize_display_text(message)
    if len(safe) <= _MAX_ERROR_MESSAGE_SNIPPET:
        return safe
    overflow = len(safe) - _MAX_ERROR_MESSAGE_SNIPPET
    return safe[:_MAX_ERROR_MESSAGE_SNIPPET] + f"... [truncated, {overflow} chars]"


def _validate_node_id(node_id: object) -> None:
    """Validate node_id: reject bool, non-int, and < 1. Node id 0 is the
    upstream "no node" sentinel, so it can never be a real member."""
    if isinstance(node_id, bool) or not isinstance(node_id, int):
        raise TypeError(f"node_id must be int, got {type(node_id).__name__}")
    if node_id < 1:
        raise ValueError(f"node_id must be >= 1, got {node_id}")


def _observe_drain_exception(t: asyncio.Task[None]) -> None:
    """Done-callback that observes a shielded drain task's exception so a
    mid-shield outer cancel doesn't leave "Task exception was never retrieved"
    at GC. Suppress narrowed to Exception so KI/SystemExit still propagate."""
    if not t.cancelled():
        with contextlib.suppress(Exception):
            t.exception()


@final
@dataclass(frozen=True, slots=True)
class LeaderInfo:
    """``(node_id, address)`` from :meth:`ClusterClient.leader_info`. Distinct
    from NodeInfo because LeaderResponse has no role field."""

    node_id: int
    address: str


@final
@dataclass(frozen=True, slots=True)
class NodeMetadata:
    """Per-node failure-domain + weight metadata from :meth:`ClusterClient.describe`."""

    failure_domain: int
    weight: int


@dataclass(frozen=True, slots=True)
class _LeaderHit:
    # A probe resolved a leader address (already redirect-policy-checked).
    address: str


@dataclass(frozen=True, slots=True)
class _ProbeMiss:
    # A probe yielded no leader. ``message`` joins the aggregate ClusterError;
    # ``exc`` chains into the BaseExceptionGroup, or None for no-leader-known.
    message: str
    exc: BaseException | None


class ClusterClient:
    """Discovers the current dqlite leader.

    :meth:`find_leader` is single-shot; :meth:`connect` wraps discovery +
    connection in bounded-backoff retry. Holds no long-lived resources
    (each probe opens a short-lived socket); the caller owns the NodeStore.
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

        ``timeout`` is the default for ``dial_timeout``/``attempt_timeout`` and
        the per-RPC budget. ``redirect_policy=None`` accepts all redirects
        (SSRF-prone); supply a callable or :func:`allowlist_policy` to constrain
        targets. ``trust_server_heartbeat`` opt-in widens the post-handshake
        read deadline (300s cap) for operator-controlled clusters only.
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
        # Split from ``timeout`` so operators can set a tight TCP-dial budget
        # and a generous attempt envelope (go-dqlite DialTimeout/AttemptTimeout).
        self._dial_timeout = dial_timeout if dial_timeout is not None else timeout
        self._attempt_timeout = attempt_timeout if attempt_timeout is not None else timeout
        self._concurrent_leader_conns = concurrent_leader_conns
        self._redirect_policy = redirect_policy
        # DoS/heartbeat governors forwarded to every admin path and leader
        # probe. Validate at construction so a misconfig surfaces at config-load
        # rather than as a per-node probe failure inside find_leader().
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        # Symmetric in/out frame cap forwarded to every admin path and leader
        # probe; ``None`` defers to the wire-layer default (64 MiB). Validate
        # at construction; reject bool (operators mean a count, not a flag).
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
        # Last-known-leader cache: set on each successful sweep, probed first
        # on the next find_leader. CPython-atomic via the GIL; a free-threaded
        # (PEP 703) port MUST add a lock before extending this beyond a str|None.
        self._last_known_leader: str | None = None
        # Single-flight slot map for find_leader: concurrent callers share the
        # in-flight sweep; slots clear on done-callback so failures aren't
        # cached. Keyed by (trust_server_heartbeat, policy) so callers with
        # different heartbeat-trust or audit-policy semantics don't collapse
        # onto each other's task (heartbeat collapse would be a security regression).
        self._find_leader_tasks: dict[tuple[bool, RedirectPolicy | None], asyncio.Task[str]] = {}
        # Fork-after-init guard: the slot map holds parent-loop-bound tasks.
        self._creator_pid = os.getpid()

    def _check_pid(self) -> None:
        """Raise ``InterfaceError`` if called in a forked child."""
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
        """Create cluster client from a list of addresses.

        ``create_pool`` does not surface ``concurrent_leader_conns``; for a
        non-default value construct a ``ClusterClient`` and pass ``cluster=``.
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
        # Holds a loop-bound slot map and a NodeStore reference; a pickled
        # duplicate would be detached from both. Surface a clear TypeError.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — holds a "
            f"loop-bound single-flight slot map and a NodeStore "
            f"reference; reconstruct from configuration in the target "
            f"process instead."
        )

    def _get_last_known_leader(self) -> str | None:
        """Return the cached last-known-leader address, or ``None``."""
        return self._last_known_leader

    def _set_last_known_leader(self, address: str | None) -> None:
        """Update the last-known-leader cache; ``None`` clears it."""
        self._last_known_leader = address

    def _check_redirect(self, address: str, *, policy: RedirectPolicy | None = None) -> None:
        """Reject leader-redirect targets that fail the policy. ``policy``
        overrides the instance default; ``None`` falls back to it."""
        effective = policy if policy is not None else self._redirect_policy
        if effective is None:
            return
        if not effective(address):
            # DEBUG so SSRF-style attempts / policy misconfigs are traceable
            # from logs, not only an exception stack.
            logger.debug("cluster: redirect rejected by policy to=%s", sanitize_for_log(address))
            # raw_message carries the verbatim peer string for cross-process
            # forensic recovery (capped by DqliteError.__init__).
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
        """Find the current cluster leader, returning its address.

        ``trust_server_heartbeat`` is forwarded for API parity but has NO
        effect on probes: ``_query_leader`` skips the full handshake, so no
        WelcomeResponse is read and the read-deadline widening never fires.

        ``policy`` overrides ``self._redirect_policy`` for this sweep (audit
        mode). Caveat: it is part of the single-flight slot key by callable
        identity, so two callers passing distinct inline lambdas do NOT
        collapse — to keep single-flight, set ``redirect_policy`` at
        construction and pass ``policy=None``.

        Concurrent callers share one in-flight sweep (single-flight); failures
        are not cached. The sweep snapshots the node store once, so a
        ``set_nodes(...)`` landing mid-sweep is only visible to the next sweep.

        Raises:
            InterfaceError: used after fork (``_check_pid`` guard).
            ClusterError: no node responded with a leader address.
            DqliteConnectionError: cached fast-path peer unreachable.
            ClusterPolicyError: redirect target rejected by policy.
        """
        self._check_pid()
        # Resolve policy=None to self._redirect_policy BEFORE building the slot
        # key so callers passing None and callers passing the same policy
        # explicitly collapse onto one task rather than running twice.
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
                # Clear only if the slot still points at THIS task; a newer
                # caller may already have supplanted us.
                if self._find_leader_tasks.get(key) is t:
                    del self._find_leader_tasks[key]
                # Observe the exception so a fully-cancelled shield set does
                # not leave "Task exception was never retrieved" at GC.
                if not t.cancelled():
                    with contextlib.suppress(Exception):
                        t.exception()

            # Register the done-callback BEFORE inserting into the slot so a
            # signal-raise between create_task and add_done_callback cannot
            # leave the slot pointing at an unobserved task.
            task.add_done_callback(_clear_slot)
            self._find_leader_tasks[key] = task
        # Shield so a caller's outer cancel does not kill the shared task; the
        # cancel still propagates to the caller via the await.
        return await asyncio.shield(task)

    async def _find_leader_impl(
        self,
        *,
        trust_server_heartbeat: bool,
        policy: RedirectPolicy | None = None,
    ) -> str:
        """Single-flight backing for ``find_leader``; see it for the contract.

        Probes nodes in parallel (bounded by ``_concurrent_leader_conns``);
        first leader-resolving probe wins, siblings are cancelled and drained.
        Per-node failures become ``_ProbeMiss`` accumulated into the aggregate
        ClusterError; ``ClusterPolicyError`` instead propagates (deterministic
        config error). Fast path: if a prior sweep cached a leader, probe it
        first and fall through on miss.
        """
        cached = self._get_last_known_leader()
        if cached is not None:
            try:
                # asyncio.timeout (cancel-scope), not wait_for, which would
                # discard the inner result on outer-cancel.
                async with asyncio.timeout(self._attempt_timeout):
                    cached_leader = await self._query_leader(
                        cached,
                        trust_server_heartbeat=trust_server_heartbeat,
                    )
                if cached_leader:
                    if not _addr_equiv(cached_leader, cached):
                        # Cached node redirected; _check_redirect may raise
                        # ClusterPolicyError (handled below).
                        self._check_redirect(cached_leader, policy=policy)
                        # Re-verify the redirect target self-identifies as
                        # leader; clear cache and fall through on mismatch.
                        # Pass attempt_timeout (not dial_timeout): this path is
                        # sequential probe+verify, not the sibling sweep shape.
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
                        else:
                            self._set_last_known_leader(cached_leader)
                            return cached_leader
                    else:
                        # Cached node confirmed itself leader. Re-validate
                        # against the policy so a tightened allowlist takes
                        # effect without waiting for the next flip.
                        self._check_redirect(cached_leader, policy=policy)
                        return cached_leader
                else:
                    # No-leader-known: leader flipped/stepped down. Clear and
                    # fall through.
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
                # Fast-path miss: clear cache and fall through. ValueError
                # translates parse_address rejection of a malformed entry.
                logger.debug(
                    "find_leader: fast-path probe of cached leader %s failed (%s); "
                    "clearing cache and falling through to full sweep",
                    sanitize_for_log(cached),
                    type(e).__name__,
                )
                self._set_last_known_leader(None)
            except ClusterPolicyError as policy_exc:
                # Cached address redirected to a policy-rejected target. Clear
                # and propagate (same policy applies to every probe). Chain via
                # `from` so __cause__ records which cached node redirected us.
                self._set_last_known_leader(None)
                raise policy_exc from ClusterError(
                    f"find_leader: fast-path probe of cached leader "
                    f"{sanitize_for_log(cached)} redirected to a "
                    f"policy-rejected target ({policy_exc})"
                )

        # Fresh mutable list: defends against a third-party NodeStore mutating
        # its returned storage mid-sweep, and lets the shuffle+sort run in place.
        nodes = list(await self._safe_node_snapshot())

        if not nodes:
            raise ClusterError("No nodes configured")

        # Shuffle (stampede-avoidance) then stable-sort by role so voters probe
        # before standbys before spares — non-voters lag and are less likely to
        # know the leader. The stable sort preserves the within-role shuffle.
        _cluster_random.shuffle(nodes)
        nodes.sort(key=lambda n: int(n.role))

        total_nodes = len(nodes)
        semaphore = asyncio.Semaphore(self._concurrent_leader_conns)

        errors: list[str] = []
        # Collect every per-node exception so the final ClusterError can chain
        # them all via BaseExceptionGroup (not just the last iteration's).
        per_node_excs: list[BaseException] = []

        async def _probe_one(idx: int, node: _StoreNodeInfo) -> _LeaderHit | _ProbeMiss:
            # Per-node probe. Returns _LeaderHit / _ProbeMiss; lets
            # ClusterPolicyError propagate to the gather loop.
            #
            # The semaphore is held ONLY across the initial _query_leader; the
            # drain and the _verify_redirect re-probe run outside the slot, so
            # holding it across the re-probe doesn't bottleneck a redirect
            # stampede. Thus the slot bounds initial probes, not verify dials.
            sem_acquired = False
            try:
                await semaphore.acquire()
                sem_acquired = True
            except (KeyboardInterrupt, SystemExit):
                # A signal-raise between acquire() returning and the
                # sem_acquired store would leak a permit (the finally guard
                # misfires). Release defensively; suppress ValueError for the
                # case the signal beat the decrement and release over-credits.
                if not sem_acquired:
                    with contextlib.suppress(ValueError):
                        semaphore.release()
                raise
            try:
                try:
                    # asyncio.timeout (cancel-scope), not wait_for.
                    async with asyncio.timeout(self._attempt_timeout):
                        leader_address = await self._query_leader(
                            node.address,
                            trust_server_heartbeat=trust_server_heartbeat,
                        )
                except TimeoutError as e:
                    # sanitize_for_log escapes LF/TAB so the address in
                    # _ProbeMiss.message can't split a downstream log (CWE-117).
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
                    # Narrow catch so programming bugs propagate. ValueError
                    # translates a peer's malformed-redirect parse_address
                    # rejection so one hostile node can't sabotage the sweep.
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
                            # sanitize_for_log: no LF/TAB into the log (CWE-117).
                            f"{sanitize_for_log(node.address)}: "
                            f"{sanitize_for_log(_truncate_error(str(e)))}"
                        ),
                        exc=e,
                    )

                # Release the slot before the verify/log work; the outer
                # finally is the safety net for the exception paths.
                if sem_acquired:
                    semaphore.release()
                    sem_acquired = False

                if leader_address:
                    # Policy gates EVERY confirmed-leader address, including the
                    # probed node's own, so an allowlist isn't bypassed when the
                    # excluded node happens to be leader. Raises on rejection;
                    # the gather loop propagates.
                    self._check_redirect(leader_address, policy=policy)
                    # Only addresses that differ from node.address are real
                    # redirects needing re-verification (canonical-tuple compare
                    # so IPv6 bracketing isn't mistaken for a redirect).
                    if not _addr_equiv(leader_address, node.address):
                        # Re-probe to confirm self-identifies as leader; a
                        # stale hint otherwise wastes a connect() round-trip.
                        verified = await self._verify_redirect(
                            leader_address,
                            trust_server_heartbeat=trust_server_heartbeat,
                        )
                        if verified is None:
                            # sanitize_for_log: server-supplied address can't
                            # split a log record (CWE-117).
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
                            return _ProbeMiss(
                                message=(f"{_safe_addr_log}: stale redirect to {_safe_hint_log}"),
                                exc=None,
                            )
                    return _LeaderHit(address=leader_address)

                # _query_leader returns None for the legitimate no-leader-known
                # reply; record it so the errors list isn't silently empty.
                _safe_addr = sanitize_for_log(node.address)
                logger.debug(
                    "find_leader: %s reports no leader known (%d/%d)",
                    _safe_addr,
                    idx + 1,
                    total_nodes,
                )
                return _ProbeMiss(message=f"{_safe_addr}: no leader known", exc=None)
            finally:
                # Safety-net release for any exception path that skipped the
                # explicit release above.
                if sem_acquired:
                    semaphore.release()

        # Build ``pending`` inside the try frame so a BaseException raised
        # mid-construction still lets the finally cancel + gather created tasks.
        pending: set[asyncio.Task[_LeaderHit | _ProbeMiss]] = set()
        winning_address: str | None = None
        policy_error: ClusterPolicyError | None = None
        unexpected_exc: BaseException | None = None
        try:
            # Create all probe tasks up-front so verifies overlap across nodes;
            # gating creation behind the semaphore would serialise them. Yield
            # periodically so a near-cap NodeStore doesn't monopolise the loop.
            for idx, n in enumerate(nodes):
                pending.add(asyncio.create_task(_probe_one(idx, n)))
                if (idx + 1) % _PROBE_TASK_CREATE_YIELD_EVERY == 0:
                    await asyncio.sleep(0)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    # Non-None means the probe let an exception propagate: only
                    # ClusterPolicyError does so intentionally; anything else is
                    # a bug, re-raised after cancelling siblings.
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
            # Cancel siblings on success / policy-error / bug / outer-cancel.
            # _query_leader's shielded finally drains each cancelled socket so
            # cancellation doesn't leak FDs; await so the drain completes here.
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if unexpected_exc is not None:
            raise unexpected_exc
        if winning_address is not None:
            if policy_error is not None:
                # A sibling redirected to a policy-rejected target before the
                # winner self-confirmed. WARN so a SIEM sees the sweep won past
                # it; sanitize_for_log guards CWE-117.
                logger.warning(
                    "find_leader: dropped policy rejection during successful "
                    "sweep — rejected redirect=%s, winning leader=%s",
                    sanitize_for_log(str(policy_error)),
                    sanitize_for_log(winning_address),
                )
            # Cache the leader so the next find_leader takes the fast path.
            self._set_last_known_leader(winning_address)
            return winning_address
        if policy_error is not None:
            # Invalidate the cache so the next call re-sweeps rather than
            # re-hitting a cached address that only redirects to a rejected one.
            self._set_last_known_leader(None)
            # Chain in per-node history so an operator can tell a policy
            # rejection on a healthy cluster from one on a half-down cluster.
            # raise ... from ... preserves the ClusterPolicyError class.
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
        # Aggregate-failure WARNING (per-node probes are DEBUG). sanitize_for_log
        # guards against a hostile FailureResponse splitting the log line.
        logger.warning(
            "cluster: leader discovery failed across %d nodes; errors=%s",
            total_nodes,
            sanitize_for_log(joined),
        )
        # Chain via BaseExceptionGroup for >1 real exception; single keeps the
        # narrow chain; no-exception (all no-leader-known) raises bare.
        if len(per_node_excs) > 1:
            raise ClusterError(f"Could not find leader. Errors: {joined}") from _bounded_group(
                "find_leader: per-node failures", per_node_excs
            )
        if per_node_excs:
            raise ClusterError(f"Could not find leader. Errors: {joined}") from per_node_excs[0]
        raise ClusterError(f"Could not find leader. Errors: {joined}")

    async def _safe_node_snapshot(self) -> tuple[_StoreNodeInfo, ...]:
        """Defensive tuple snapshot of the node store, bounded by dial_timeout.

        Copies so a third-party store mutating its returned storage can't tear
        the sweep; the timeout stops a blocking-I/O store from pinning
        find_leader. Use everywhere instead of get_nodes() directly.
        """
        async with asyncio.timeout(self._dial_timeout):
            return tuple(await self._node_store.get_nodes())

    async def _query_leader(
        self, address: str, *, trust_server_heartbeat: bool = False
    ) -> str | None:
        """Query a node for the current leader.

        Raises OSError on dial failure, DqliteConnectionError/ProtocolError/
        OperationalError on handshake-or-later failure (the caller attributes
        each class per node). Returns None when reachable but no leader known.
        Transport errors propagate (rather than collapsing to None) so the
        aggregate can distinguish "unreachable" from "no leader elected".
        """
        # writer=None before the try, dial INSIDE it, so a cancel between
        # dial-success and protocol-construction drains the writer in finally
        # rather than orphaning it. asyncio.timeout (cancel-scope), not wait_for.
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
            # Probe-only handshake: skip ClientRequest registration (the C
            # server's handle_leader doesn't need client_id), saving an RTT and
            # a server slot per probe. The chosen leader gets full registration.
            await protocol.negotiate_protocol_only()
            node_id, leader_addr = await protocol.get_leader()

            # (id=N, address="") is a RAFT_NOMEM transient: recvUpdateLeader
            # set the id but failed to malloc the address. Treat as "leader
            # unknown" (as Go/C clients do) rather than raising a ProtocolError.
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
                # Inverse illegal shape: raft_leader never writes a non-empty
                # address with id=0, so reject rather than trust the redirect.
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
            return None
        finally:
            # writer is None if the dial failed or cancel landed before unpack;
            # gate the drain on its presence (a `return` in finally would
            # silently swallow a propagating exception).
            if writer is not None:
                writer.close()

                # Bounded wait_closed so the transport doesn't sit in FIN-WAIT
                # under leader-probe churn. Shield + an explicit observed Task
                # so an outer cancel mid-drain neither orphans the reader task
                # nor leaves an unobserved-exception warning at GC. The shielded
                # drain runs its 100 ms tail even during outer-cancel shutdown.
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
        """Re-probe a redirect target to confirm it self-identifies as leader.

        Returns ``hint_address`` on self-confirmation, ``None`` on mismatch /
        unreachable / any transport failure (verification failure is not fatal;
        the sweep falls through). A stale peer can hand back a node that no
        longer leads, which would otherwise cost a wasted connect()+Open.
        Delegates to ``_query_leader`` so transport discipline and mocking
        apply uniformly. ``timeout`` defaults to ``self._dial_timeout``; since
        the verify runs sibling to the initial probe, the per-probe budget is
        ``attempt_timeout + dial_timeout``.
        """
        effective_timeout = self._dial_timeout if timeout is None else timeout
        try:
            # asyncio.timeout (cancel-scope), not wait_for, which would discard
            # the verified address on outer-cancel and defeat the cache.
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
            # ValueError covers parse_address rejecting a malformed hint; map
            # to None so the sweep falls through like other verify failures.
            return None
        if reported and _addr_equiv(reported, hint_address):
            return hint_address
        # Stale or pointing elsewhere. sanitize_for_log guards CWE-117.
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
        """Connect to the cluster leader, with bounded-backoff retry.

        ``max_attempts`` defaults to 3 (one leader change + one transport
        hiccup). ``max_elapsed_seconds`` is an optional wall-clock cap; it must
        be passed explicitly because there is no implicit infinite retry.
        Per-iteration backoff is capped at 1 s; each failure logs at DEBUG.

        Two divergences from go-dqlite's Connect for porters:
        1. asyncio cancel propagates through every await, so an outer
           timeout(0) lands BEFORE the first attempt (Go always runs one).
        2. Discovery returns an address and re-dials, so each successful
           connect pays one extra handshake RTT to the leader (the
           single-flight slot amortises the sweep across concurrent callers).

        ``policy`` overrides the instance ``redirect_policy`` for this connect
        only; passing one defeats find_leader's single-flight collapse.
        """
        # Reject bool before the < 1 check so True/False don't coerce to 1/0.
        if max_attempts is not None and (
            isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
        ):
            raise TypeError(f"max_attempts must be int or None, got {type(max_attempts).__name__}")
        attempts_cap = max_attempts if max_attempts is not None else _DEFAULT_CONNECT_MAX_ATTEMPTS
        if attempts_cap < 1:
            raise ValueError(f"max_attempts must be at least 1 if provided, got {attempts_cap}")
        # Duplicate retry.py's validation here so a misconfig surfaces at the
        # connect() call site, not after attempt 1 burns its budget. Wording
        # matches retry.py exactly so message-pinning tests stay green.
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

        # Per-call overrides instance default; None falls back. Without this,
        # the kwarg's None default would silently drop the constructor's
        # max_message_size cap on the connect path.
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
                # Pre-validate the address so "invalid leader address" fires
                # only on the server's redirect; constructor ValueErrors
                # (close_timeout, etc.) propagate with their own message rather
                # than being misattributed as a server-redirect failure.
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
                    # Explicitly close the half-built conn so its scheduled
                    # _pending_drain task isn't GC'd mid-flight ("Task was
                    # destroyed but it is pending"). Absorb CancelledError, log
                    # transport teardown failures, let KI/SystemExit/bugs
                    # propagate; the bare raise re-delivers the original error.
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
                # Narrow catch: the transport/cluster failures the retry loop
                # re-attempts (wider would mask programming bugs). ValueError is
                # a backstop for a malformed-address rejection escaping the sweep
                # translation; OperationalError is here only for the DEBUG
                # breadcrumb (it stays non-retryable below).
                #
                # Invalidate the cache so the next retry re-sweeps rather than
                # probing the leader we just failed against. Gated on leader is
                # not None (find_leader already clears on its own failure).
                if leader is not None:
                    self._set_last_known_leader(None)
                logger.debug(
                    "ClusterClient.connect attempt %d/%d failed (leader=%r): %s",
                    attempt,
                    attempts_cap,
                    leader,
                    # sanitize_for_log: server text can't split the record (CWE-117).
                    sanitize_for_log(_truncate_error(str(exc))),
                )
                raise

        # Retry only transport-level errors. Leader-change OperationalErrors are
        # already reclassified to DqliteConnectionError, so a schema/SQL error
        # isn't amplified into 5×N RTTs. ClusterPolicyError is excluded
        # (deterministic; retrying just reproduces it).
        try:
            return await retry_with_backoff(
                try_connect,
                max_attempts=attempts_cap,
                max_delay=_DEFAULT_CONNECT_MAX_DELAY,
                max_elapsed_seconds=max_elapsed_seconds,
                # OSError subsumes every stdlib transport-error shape.
                retryable_exceptions=(
                    DqliteConnectionError,
                    ClusterError,
                    OSError,
                ),
                excluded_exceptions=(ClusterPolicyError,),
            )
        except ClusterPolicyError:
            # Excluded from retry; caught separately (it's a ClusterError
            # subclass) so the handler below doesn't misreport a single
            # deterministic rejection as N exhausted attempts.
            raise
        except (DqliteConnectionError, ClusterError, OSError) as exc:
            # Aggregate-failure WARNING (per-attempt failures are DEBUG).
            # sanitize_for_log guards CWE-117.
            logger.warning(
                "cluster: connect exhausted %d attempts; last_error=%s: %s",
                attempts_cap,
                type(exc).__name__,
                sanitize_for_log(_truncate_error(str(exc))),
            )
            raise

    async def cluster_info(self, *, policy: RedirectPolicy | None = None) -> list[_WireNodeInfo]:
        """Return the current cluster's node list (id, address, role).

        Asks the leader (any node could answer, but the leader's view is
        freshest), re-confirming leadership with one extra round-trip so a
        mid-RPC flip can't feed a stale config into ``set_nodes(...)``.

        ``policy`` filters returned addresses (rejected ones are dropped with a
        warning), guarding against a hostile leader smuggling addresses into a
        membership rotation. ``None`` falls back to ``self._redirect_policy``;
        pass an always-True callable to disable filtering.

        Raises:
            InterfaceError: used after fork.
            DqliteConnectionError: admin connection could not be established.
            ClusterError: no leader reachable.
            OperationalError: leader rejected the request.
            ProtocolError: wire-level shape mismatch.
        """
        try:
            leader_addr = await self.find_leader(policy=policy)
            async with self.open_admin_connection(leader_addr) as protocol:
                # Re-confirm leadership before reading the config: a flip
                # between find_leader and now could leave us reading from a
                # follower whose view strips the real leader.
                node_id, address = await protocol.get_leader()
                # Empty address ((0,"") sentinel or (N,"") RAFT_NOMEM) means
                # "leader not known": skip the redirect-chase and read from the
                # current responder, matching Go's len(address)==0 check.
                if not _addr_equiv(address, leader_addr) and address:
                    # Leadership flipped: re-validate against policy and re-probe
                    # before reading the config from the new address.
                    self._check_redirect(address, policy=policy)
                    verified = await self._verify_redirect(
                        address,
                        trust_server_heartbeat=False,
                    )
                    if verified is None:
                        # Stale hint that did not re-confirm: surface a
                        # ClusterError rather than the stale responder's view.
                        self._set_last_known_leader(None)
                        raise ClusterError(
                            "cluster_info: leadership flipped mid-RPC "
                            "and the responder's hint did not re-confirm"
                        )
                    async with self.open_admin_connection(verified) as p2:
                        nodes = await p2.cluster()
                    # Cache the verified leader so the next find_leader hits the
                    # fast path on the post-flip scenario where it matters most.
                    self._set_last_known_leader(verified)
                else:
                    nodes = await protocol.cluster()
        except (OperationalError, DqliteConnectionError, ProtocolError):
            # Failure-path invalidation only: a mid-RPC step-down surfaces as
            # one of these. On success the leader just answered, so the cache
            # stays warm for the next find_leader fast path.
            self._set_last_known_leader(None)
            raise
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
                    # sanitize_for_log: server address can't split the record (CWE-117).
                    sanitize_for_log(node.address),
                    node.node_id,
                    node.role.name,
                )
            # Yield periodically so a near-cap (10_000) node list whose policy
            # check is per-node doesn't monopolise the loop.
            if (i + 1) % _PROBE_TASK_CREATE_YIELD_EVERY == 0:
                await asyncio.sleep(0)
        return filtered

    async def transfer_leadership(self, target_node_id: int) -> None:
        """Transfer leadership to ``target_node_id`` (must be a reachable voter).

        Ops-grade primitive (not for SQL-path use). Returns once the server
        accepts; election convergence is observable via a later find_leader.

        Raises:
            InterfaceError: used after fork.
            ClusterError: no leader reachable.
            OperationalError: server rejected (target not a voter, etc.).
            ProtocolError: wire-level shape mismatch.
        """
        # Validate locally so callers see a clean TypeError, not a wire-decode error.
        if isinstance(target_node_id, bool) or not isinstance(target_node_id, int):
            raise TypeError(f"target_node_id must be int, got {type(target_node_id).__name__}")
        if target_node_id < 1:
            raise ValueError(f"target_node_id must be >= 1, got {target_node_id}")

        # No pre-RPC invalidation: the find_leader fast path already recovers a
        # stepped-down cache in one extra probe, and pre-invalidating would
        # force a full sweep on every transfer (including warm-cached no-ops).
        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.transfer(target_node_id)
        finally:
            # Leader stepped down (success) or RPC failed (possible flip): either
            # way the cache is suspect, so invalidate on both paths.
            self._set_last_known_leader(None)

    async def leader_info(self, *, policy: RedirectPolicy | None = None) -> LeaderInfo | None:
        """Return the current leader's ``(node_id, address)``, or ``None``.

        Cheaper than :meth:`cluster_info` when only the leader id is needed.
        ``None`` is the legitimate "no leader yet" reply during re-election.

        ``policy`` gates the responder's self-reported address on a flip between
        find_leader and the follow-up get_leader, so a hostile follower can't
        tunnel an attacker-controlled address. Falls back to
        ``self._redirect_policy``; pass an always-True callable to disable.

        Raises:
            InterfaceError: used after fork.
            ClusterError: no node responded.
            ClusterPolicyError: a flipped responder's address fails the policy.
        """
        # Reuse find_leader's sweep, then ask the leader its node id.
        try:
            leader_addr = await self.find_leader(policy=policy)
            async with self.open_admin_connection(leader_addr) as protocol:
                node_id, address = await protocol.get_leader()
                if node_id == 0 and not address:
                    # Mid-election: responder no longer self-identifies as leader.
                    return None
                if node_id != 0 and not address:
                    # RAFT_NOMEM transient (id set, address malloc failed):
                    # treat as "no leader known" like _query_leader's arm.
                    logger.debug(
                        "leader_info: %s reports leader_id=%d with empty "
                        "address (RAFT_NOMEM transient — treating as "
                        "'no leader known')",
                        sanitize_for_log(leader_addr),
                        node_id,
                    )
                    return None
                if node_id == 0 and address:
                    # Malformed (0, nonempty): raft_leader never emits it, so a
                    # hostile follower can't slip it past the policy gate as None.
                    raise ProtocolError(
                        f"leader_info: malformed (node_id, address) — "
                        f"got id={node_id!r} addr={_sanitize_display_text(address)!r}; "
                        f"node_id=0 must be paired with an empty address"
                    )
                if not _addr_equiv(address, leader_addr):
                    # Leadership flipped: re-validate against policy (so a
                    # hostile follower can't tunnel an address) and re-probe to
                    # confirm self-identifies as leader before using its id.
                    self._check_redirect(address, policy=policy)
                    verified = await self._verify_redirect(
                        address,
                        trust_server_heartbeat=False,
                    )
                    if verified is None:
                        # Stale hint that did not re-confirm: "no leader known".
                        return None
                    # Re-fetch from the verified responder so the returned
                    # LeaderInfo's node_id is internally consistent.
                    async with self.open_admin_connection(verified) as p2:
                        vnode_id, vaddress = await p2.get_leader()
                    if vnode_id == 0 and not vaddress:
                        return None
                    if vnode_id != 0 and not vaddress:
                        # RAFT_NOMEM transient on the verified hint; "no leader known".
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
                    # Gate the third-hop address too: the verified responder is
                    # allowlisted but its reported leader can be any peer.
                    self._check_redirect(vaddress, policy=policy)
                    return LeaderInfo(node_id=vnode_id, address=vaddress)
                return LeaderInfo(node_id=node_id, address=address)
        except (OperationalError, DqliteConnectionError, ProtocolError):
            # Failure-path invalidation only; on success the cache stays warm.
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

        Two-phase like go-dqlite's Client.Add: ADD always lands the node as
        SPARE; a non-spare role triggers a follow-up ASSIGN on the same
        connection. Not atomic — if ADD succeeds but ASSIGN raises, the node is
        left SPARE; recover with an idempotent ``assign_role(node_id, role)``
        rather than retrying ADD (which would trip "node already exists").

        Raises:
            InterfaceError: used after fork.
            TypeError / ValueError: invalid arguments.
            ClusterError: no leader reachable.
            OperationalError: server rejected (may be from ADD or ASSIGN).
            ProtocolError: wire-level shape mismatch.
        """
        _validate_node_id(node_id)
        if not isinstance(address, str) or not address:
            raise TypeError(f"address must be a non-empty str, got {type(address).__name__}")
        if not isinstance(role, NodeRole):
            raise TypeError(f"role must be a NodeRole, got {type(role).__name__}")
        # Validate shape now: an unvalidated malformed address would be stored
        # in the Raft log and only fail later when a node tries to dial it.
        try:
            parse_address(address)
        except ValueError as exc:
            raise ValueError(f"add_node: invalid address {address!r}: {exc}") from exc

        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.add(node_id, address)
                if role != NodeRole.SPARE:
                    # Second phase: promote via Assign on the same connection so
                    # both requests land on the same leader.
                    await protocol.assign(node_id, role)
        finally:
            # Membership changes interleave with elections; invalidate on both paths.
            self._set_last_known_leader(None)

    async def assign_role(self, node_id: int, role: NodeRole) -> None:
        """Change a node's role (promote a caught-up spare, or demote a voter).

        Raises:
            InterfaceError: used after fork.
            TypeError / ValueError: invalid arguments.
            ClusterError: no leader reachable.
            OperationalError: server rejected.
            ProtocolError: wire-level shape mismatch.
        """
        _validate_node_id(node_id)
        if not isinstance(role, NodeRole):
            raise TypeError(f"role must be a NodeRole, got {type(role).__name__}")

        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.assign(node_id, role)
        finally:
            # Promotion widens the voting set (election window); invalidate on
            # both paths.
            self._set_last_known_leader(None)

    async def remove_node(self, node_id: int) -> None:
        """Remove a node from the cluster (Raft membership change).

        Removing the current leader requires a prior :meth:`transfer_leadership`
        to another voter, else the server rejects with a not-leader error.

        Raises:
            InterfaceError: used after fork.
            TypeError / ValueError: invalid arguments.
            ClusterError: no leader reachable.
            OperationalError: server rejected.
            ProtocolError: wire-level shape mismatch.
        """
        _validate_node_id(node_id)

        leader_addr = await self.find_leader()
        try:
            async with self.open_admin_connection(leader_addr) as protocol:
                await protocol.remove(node_id)
        finally:
            # The removed node may have been the cached leader; invalidate
            # unconditionally as the conservative choice.
            self._set_last_known_leader(None)

    async def describe(self, *, address: str | None = None) -> NodeMetadata:
        """Read a node's failure-domain + weight metadata.

        Per-node: returns the connected peer's own metadata. Pass ``address``
        for a specific node; ``None`` describes the leader.

        Raises:
            InterfaceError: used after fork.
            DqliteConnectionError: admin connection could not be established.
            ClusterError: no leader reachable (only when ``address=None``).
            OperationalError: node rejected the request.
            ProtocolError: wire-level shape mismatch.
        """
        # Only the address-None arm routes through the leader, so only it
        # invalidates the cache. The pid guard fires here so the per-address
        # arm carries the same fork discipline as the leader-routed sibling.
        self._check_pid()
        if address is not None:
            # Validate shape at the call site (like add_node) so a malformed
            # address surfaces as a clean ValueError, not a deep dial failure.
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
                # Failure-path invalidation only; on success the cache stays warm.
                self._set_last_known_leader(None)
            raise

    async def set_weight(self, weight: int, *, address: str | None = None) -> None:
        """Set a node's weight (leader-election preference).

        Per-node: pass ``address`` for a specific node, or ``None`` for the leader.

        Raises:
            InterfaceError: used after fork.
            TypeError / ValueError: invalid arguments.
            DqliteConnectionError: admin connection could not be established.
            ClusterError: ``address=None`` and no leader reachable.
            OperationalError: node rejected.
            ProtocolError: wire-level shape mismatch.
        """
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise TypeError(f"weight must be int, got {type(weight).__name__}")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")

        # Per-address fork guard (like describe): the explicit-address arm would
        # otherwise silently succeed in a forked child.
        self._check_pid()
        if address is not None:
            # Validate shape at the call site (like add_node / describe).
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
                # Failure-path invalidation only; on success the cache stays warm.
                self._set_last_known_leader(None)
            raise

    async def dump(self, database: str) -> dict[str, bytes]:
        """Dump a database to ``{filename: bytes}``.

        ``database`` is required (no default) since dumping the wrong database
        for a backup is a footgun. Sent to the leader, though any member could
        serve it (handle_dump does not CHECK_LEADER); back up from a spare via
        :meth:`open_admin_connection` to avoid leader-CPU contention. The wire
        layer caps file count / size, so multi-GB databases fail with
        ProtocolError — plan a cluster-side snapshot for those.

        Raises:
            InterfaceError: used after fork.
            TypeError: invalid arguments.
            DqliteConnectionError: admin connection could not be established.
            ClusterError: no leader reachable.
            OperationalError: server rejected (e.g. unknown database).
            ProtocolError: wire-level shape mismatch.
        """
        if not isinstance(database, str) or not database:
            raise TypeError(f"database must be a non-empty str, got {type(database).__name__}")

        try:
            leader_addr = await self.find_leader()
            async with self.open_admin_connection(leader_addr) as protocol:
                return await protocol.dump(database)
        except (OperationalError, DqliteConnectionError, ProtocolError):
            # Failure-path invalidation only (a step-down mid-dump is plausible);
            # on success the cache stays warm.
            self._set_last_known_leader(None)
            raise

    @contextlib.asynccontextmanager
    async def open_admin_connection(self, address: str) -> AsyncIterator[DqliteProtocol]:
        """Open a one-shot admin connection to ``address``, yield a handshaken
        :class:`DqliteProtocol`, and tear the socket down on exit.

        Public direct-to-node primitive bypassing leader discovery; used by
        every admin method here. Each call opens a fresh socket (no pool) so
        admin traffic doesn't mix with SQL-path lifecycle. ``dial_timeout``
        bounds the TCP phase; ``timeout`` is the per-RPC budget.

        Raises:
            InterfaceError: used after fork.
            DqliteConnectionError: dial or handshake failed.
            OperationalError: node rejected the handshake.
            ProtocolError: wire-level shape mismatch.
        """
        # Wrap dial+handshake in attempt_timeout so a slow-handshaking peer
        # can't hang admin RPCs for the full per-RPC timeout. Fork guard fires
        # before invoking a possibly parent-loop-bound dial_func.
        self._check_pid()
        protocol: DqliteProtocol | None = None
        writer = None
        try:
            try:
                async with asyncio.timeout(self._attempt_timeout):
                    # asyncio.timeout (cancel-scope), not wait_for, which would
                    # orphan (reader, writer) on outer-cancel between dial and unpack.
                    try:
                        async with asyncio.timeout(self._dial_timeout):
                            reader, writer = await open_connection(
                                address, dial_func=self._dial_func
                            )
                    except TimeoutError as e:
                        # Dial-specific timeout, re-raised distinctly so the
                        # outer arm can tell handshake-stall from dial-stall.
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
                    # Do NOT null writer after the hand-off (unlike
                    # _connect_impl): the protocol is yielded, not stored, so the
                    # outer finally is the only place it can be closed.
                    #
                    # Speak only the 8-byte version: no admin RPC reads
                    # client_id, so ClientRequest registration would be pure
                    # waste (an RTT + a retained server slot).
                    await protocol.negotiate_protocol_only()
            except TimeoutError as e:
                # Outer attempt_timeout fired after the dial: handshake stalled.
                # Distinct message so dial- and handshake-stall don't conflate.
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

                # Bounded, shielded, observed drain — see _query_leader.
                async def _drain() -> None:
                    async with asyncio.timeout(_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS):
                        await writer.wait_closed()

                inner_drain: asyncio.Task[None] = asyncio.ensure_future(_drain())
                inner_drain.add_done_callback(_observe_drain_exception)
                with contextlib.suppress(OSError, TimeoutError):
                    await asyncio.shield(inner_drain)


def allowlist_policy(addresses: Iterable[str]) -> RedirectPolicy:
    """Build a redirect policy accepting only the given addresses.

    Compared as ``(host, port)`` tuples, so IPv6 bracketing and host casing
    don't matter; use a custom callable for CIDR/DNS/wildcard. Entries are
    parsed at construction (typos raise ValueError here); a malformed runtime
    target returns False. The iterable is materialized once, so generators are safe.
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

    Rejects link-local (the metadata-endpoint SSRF target) unconditionally and
    loopback by default; ``include_rfc1918=True`` (default) admits private
    subnets where most clusters live. Malformed input returns False.

    SSRF caveats: applied to the literal address before resolution, so DNS
    hostnames pass through (a name resolving to a metadata IP is NOT blocked)
    — pair with :func:`allowlist_policy` for strict defense. ::ffff/6to4/Teredo
    IPv4 wrappings are unwrapped and classified; other tunnel modes are not.
    """
    import ipaddress

    def policy(addr: str) -> bool:
        try:
            host, _port = parse_address(addr)
        except ValueError:
            return False
        # Literal IPs get the address-class filter; hostnames pass through.
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True  # hostname; rely on DNS / allowlist for finer control
        # Unwrap IPv6-encapsulated IPv4 so a wrapped metadata IP classifies as
        # the embedded v4, not the (benign-looking) wrapper.
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            elif ip.sixtofour is not None:
                ip = ip.sixtofour
            elif ip.teredo is not None:
                # teredo returns (server_v4, client_v4); classify the client.
                _server_v4, client_v4 = ip.teredo
                ip = client_v4
        # Link-local rejected unconditionally (metadata-endpoint exfil target).
        if ip.is_link_local:
            return False
        if ip.is_loopback and not include_loopback:
            return False
        return not (ip.is_private and not include_rfc1918)

    return policy
