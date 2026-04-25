"""Cluster management and leader detection for dqlite."""

import asyncio
import contextlib
import logging
import random
from collections.abc import Callable, Iterable

from dqliteclient.connection import DqliteConnection, _parse_address, _validate_timeout
from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DqliteConnectionError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeStore
from dqliteclient.protocol import DqliteProtocol
from dqliteclient.retry import retry_with_backoff
from dqlitewire import NodeRole
from dqlitewire.messages.responses import _sanitize_server_text as _sanitize_display_text

__all__ = ["ClusterClient", "RedirectPolicy", "allowlist_policy"]

logger = logging.getLogger(__name__)

# Type alias for a redirect-target policy. Returns True if the address
# should be accepted, False to reject with a ClusterError.
RedirectPolicy = Callable[[str], bool]

# Default attempt count for connect(). Three attempts cover one leader
# change plus one transport hiccup; substantially higher counts risk
# hiding genuine cluster instability under what looks like "a slow
# connect". Operators can override via ClusterClient.connect(
# max_attempts=...).
_DEFAULT_CONNECT_MAX_ATTEMPTS = 3

# Cap per-node error messages at this length before concatenating them
# into the final ClusterError. A failing peer that returns a multi-MB
# FailureResponse message would otherwise produce an O(N * M) string
# held in memory and serialised into every traceback.
_MAX_ERROR_MESSAGE_SNIPPET = 200

# Budget for the bounded writer-drain in ``_query_leader``. A
# responsive peer drains FIN/ACK in microseconds; a slow peer must not
# hold up leader discovery. 100 ms is generous for LAN and still
# negligible against the per-probe ``self._timeout`` (typically seconds).
_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS = 0.1


def _truncate_error(message: str) -> str:
    if len(message) <= _MAX_ERROR_MESSAGE_SNIPPET:
        return message
    return message[:_MAX_ERROR_MESSAGE_SNIPPET] + f"... [truncated, {len(message)} chars]"


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
        timeout: float = 10.0,
        redirect_policy: RedirectPolicy | None = None,
    ) -> None:
        """Initialize cluster client.

        Args:
            node_store: Store for cluster node information
            timeout: Connection timeout in seconds
            redirect_policy: Optional callable ``(address) -> bool`` that
                authorizes each leader redirect target. If None (default),
                redirects are accepted — this preserves backward
                compatibility but permits a compromised peer to redirect
                clients to arbitrary hosts (SSRF-style). Supply a
                callable or the :func:`allowlist_policy` helper to
                constrain where redirects can go.
        """
        _validate_timeout(timeout)
        self._node_store = node_store
        self._timeout = timeout
        self._redirect_policy = redirect_policy
        # Single-flight slot map for ``find_leader``. Concurrent callers
        # share the in-flight discovery task instead of each launching
        # an independent per-node sweep — under a leader flip with N
        # waiting acquirers, this collapses N×M handshake attempts
        # against the failing ex-leader into M (one sweep). See
        # ISSUE-631 for the stampede shape this closes. Slots are
        # cleared on done-callback so a fresh probe re-runs after the
        # current task finishes; consecutive callers do NOT share a
        # cached failure.
        #
        # The map is keyed by ``(trust_server_heartbeat,)``: callers
        # that requested different heartbeat-trust semantics must NOT
        # collapse onto the same task — the per-call flag would
        # otherwise be silently ignored for the second caller, which
        # is a security-adjacent regression for operators who opted
        # one pool out of widened heartbeats. Bounded to two slots
        # since the flag is a bool.
        self._find_leader_tasks: dict[tuple[bool], asyncio.Task[str]] = {}

    @classmethod
    def from_addresses(
        cls,
        addresses: list[str],
        timeout: float = 10.0,
        *,
        redirect_policy: RedirectPolicy | None = None,
    ) -> "ClusterClient":
        """Create cluster client from list of addresses."""
        store = MemoryNodeStore(addresses)
        return cls(store, timeout=timeout, redirect_policy=redirect_policy)

    def _check_redirect(self, address: str) -> None:
        """Reject leader-redirect targets that fail the configured policy."""
        if self._redirect_policy is None:
            return
        if not self._redirect_policy(address):
            # Security-adjacent event: an operator-supplied policy
            # just rejected a server-advised leader target. Surface
            # at DEBUG so SSRF-style attempts or policy
            # misconfigurations are traceable from logs alone,
            # not only through an exception stack.
            logger.debug("cluster: redirect rejected by policy to=%s", address)
            raise ClusterPolicyError(f"Leader redirect to {address!r} rejected by redirect_policy")

    async def find_leader(self, *, trust_server_heartbeat: bool = False) -> str:
        """Find the current cluster leader.

        Returns the leader address. ``trust_server_heartbeat`` is forwarded
        to each probe protocol so operators who opted into a widened
        heartbeat window for the main query path get the same semantics
        during leader discovery.

        Concurrent callers share an in-flight discovery task (single-
        flight). Under a leader flip with N waiting acquirers, this
        collapses N independent per-node sweeps into one. Failures are
        not cached: once the current task completes, the slot clears
        so the next caller runs a fresh probe.
        """
        key: tuple[bool] = (trust_server_heartbeat,)
        task = self._find_leader_tasks.get(key)
        if task is None or task.done():
            task = asyncio.create_task(
                self._find_leader_impl(trust_server_heartbeat=trust_server_heartbeat)
            )
            self._find_leader_tasks[key] = task

            def _clear_slot(t: asyncio.Task[str]) -> None:
                # Clear the slot only if it still points at THIS task —
                # a concurrent ``find_leader`` may have already
                # supplanted us if our task finished and a new caller
                # triggered a fresh probe before this callback ran.
                if self._find_leader_tasks.get(key) is t:
                    del self._find_leader_tasks[key]

            task.add_done_callback(_clear_slot)
        # Shield so a caller's outer cancel does not kill the shared
        # task; the cancel still propagates to the calling coroutine
        # via ``await asyncio.shield``.
        return await asyncio.shield(task)

    async def _find_leader_impl(self, *, trust_server_heartbeat: bool) -> str:
        """Perform the actual leader discovery sweep. See ``find_leader``
        for the public contract — this method is the single-flight
        backing implementation."""
        nodes = await self._node_store.get_nodes()

        if not nodes:
            raise ClusterError("No nodes configured")

        # Shuffle first so repeated callers don't stampede the same node;
        # then stable-sort by role so voters come before non-voters.
        # Standby/spare nodes can never become leader (their LEADER
        # response is always (0, "")), so probing them first wastes RTTs.
        #
        # Deliberate divergence from go-dqlite: the Go connector iterates
        # nodes in their stored order (deterministic) and relies on role
        # to decide candidacy. We shuffle within role class to avoid
        # stampeding a single node across parallel callers. Do not
        # "fix" this toward Go's deterministic behavior without adding
        # an explicit stampede-avoidance mechanism elsewhere.
        nodes = list(nodes)
        random.shuffle(nodes)
        nodes.sort(key=lambda n: 0 if n.role == NodeRole.VOTER else 1)

        errors: list[str] = []
        last_exc: BaseException | None = None
        total_nodes = len(nodes)

        for idx, node in enumerate(nodes):
            try:
                leader_address = await asyncio.wait_for(
                    self._query_leader(
                        node.address,
                        trust_server_heartbeat=trust_server_heartbeat,
                    ),
                    timeout=self._timeout,
                )
                if leader_address:
                    # Only leader_address values that did NOT come from
                    # node.address itself need authorizing — those are
                    # real redirects. If the server returned its own
                    # address, it's the leader and already in the store.
                    if leader_address != node.address:
                        self._check_redirect(leader_address)
                    return leader_address
            except TimeoutError as e:
                logger.debug(
                    "find_leader: %s timed out after %.3fs (%d/%d)",
                    node.address,
                    self._timeout,
                    idx + 1,
                    total_nodes,
                )
                errors.append(f"{node.address}: timed out")
                last_exc = e
                continue
            except (DqliteConnectionError, ProtocolError, OperationalError, OSError) as e:
                # Narrow the catch so programming bugs (TypeError, KeyError,
                # etc.) propagate directly instead of being stringified into
                # a retryable ClusterError.
                logger.debug(
                    "find_leader: %s failed with %s: %s (%d/%d)",
                    node.address,
                    type(e).__name__,
                    _truncate_error(str(e)),
                    idx + 1,
                    total_nodes,
                )
                errors.append(f"{node.address}: {_truncate_error(str(e))}")
                last_exc = e
                continue

        raise ClusterError(f"Could not find leader. Errors: {'; '.join(errors)}") from last_exc

    async def _query_leader(
        self, address: str, *, trust_server_heartbeat: bool = False
    ) -> str | None:
        """Query a node for the current leader."""
        host, port = _parse_address(address)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._timeout,
            )
        except OSError:
            # OSError subsumes TimeoutError, BrokenPipeError,
            # ConnectionError, ConnectionRefusedError, and the rest
            # of the stdlib transport-error shapes. Any one of those
            # here means the node is unreachable; surface "unknown
            # leader" to the caller so it can try another node.
            return None

        try:
            protocol = DqliteProtocol(
                reader,
                writer,
                timeout=self._timeout,
                trust_server_heartbeat=trust_server_heartbeat,
                address=address,
            )
            await protocol.handshake()
            node_id, leader_addr = await protocol.get_leader()

            # Upstream dqlite's ``raft_leader`` sets ``id`` and ``address``
            # atomically: either both are filled in (a leader is known) or
            # both are zero/NULL. The server substitutes ``""`` for NULL,
            # so the only wire-legal shapes are ``(0, "")`` and
            # ``(nonzero, nonempty)``.
            if node_id != 0 and not leader_addr:
                # Observability: an operator tailing logs during a
                # leader-discovery stall needs the breadcrumb at the
                # point of detection, not only via the raised
                # ProtocolError surfaced upstream. Log at DEBUG so the
                # WARN/ERROR paths stay uncluttered during healthy
                # per-node probes.
                logger.debug(
                    "query_leader: %s returned malformed redirect (node_id=%s, address=%r)",
                    address,
                    node_id,
                    leader_addr,
                )
                raise ProtocolError(
                    f"server {_sanitize_display_text(address)} returned "
                    f"node_id={node_id} with empty leader address; "
                    f"expected both or neither"
                )
            if node_id == 0 and leader_addr:
                # Mirror arm: the inverse illegal shape. Upstream
                # ``raft_leader`` never writes a non-empty address with
                # id=0, so a peer returning this is either confused or
                # hostile. Reject symmetrically so the redirect target
                # is not trusted without a matching id.
                logger.debug(
                    "query_leader: %s returned malformed redirect (node_id=0, address=%r)",
                    address,
                    leader_addr,
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
            with contextlib.suppress(OSError, asyncio.TimeoutError):
                await asyncio.shield(
                    asyncio.wait_for(
                        writer.wait_closed(),
                        timeout=_LEADER_PROBE_DRAIN_TIMEOUT_SECONDS,
                    )
                )

    async def connect(
        self,
        database: str = "default",
        *,
        max_total_rows: int | None = 10_000_000,
        max_continuation_frames: int | None = 100_000,
        trust_server_heartbeat: bool = False,
        close_timeout: float = 0.5,
        max_attempts: int | None = None,
    ) -> DqliteConnection:
        """Connect to the cluster leader.

        Returns a connection to the current leader. ``max_total_rows``,
        ``max_continuation_frames``, and ``trust_server_heartbeat`` are
        forwarded to the underlying :class:`DqliteConnection` so callers
        (including :class:`ConnectionPool`) can tune security/DoS
        governors from one place.

        ``max_attempts`` overrides the default
        :data:`_DEFAULT_CONNECT_MAX_ATTEMPTS`.

        Each attempt's failure is logged at DEBUG level with the
        attempted leader address and the error, so operators can
        enable debug logging to diagnose cluster churn instead of
        seeing only the final exception.

        Args:
            close_timeout: Budget (seconds) for the transport-drain
                during ``close()``. After ``writer.close()`` the
                local side of the socket is gone; ``wait_closed`` is
                best-effort cleanup. The 0.5s default is sized for
                LAN; increase for WAN deployments where FIN/ACK
                round-trip is slower, or decrease to tighten
                SIGTERM-shutdown budgets. See
                ``DqliteConnection.__init__`` for full rationale.
        """
        attempts_cap = max_attempts if max_attempts is not None else _DEFAULT_CONNECT_MAX_ATTEMPTS
        if attempts_cap < 1:
            raise ValueError(f"max_attempts must be >= 1, got {attempts_cap}")

        attempt_counter = [0]

        async def try_connect() -> DqliteConnection:
            attempt_counter[0] += 1
            attempt = attempt_counter[0]
            leader: str | None = None
            try:
                leader = await self.find_leader(
                    trust_server_heartbeat=trust_server_heartbeat,
                )
                conn = DqliteConnection(
                    leader,
                    database=database,
                    timeout=self._timeout,
                    max_total_rows=max_total_rows,
                    max_continuation_frames=max_continuation_frames,
                    trust_server_heartbeat=trust_server_heartbeat,
                    close_timeout=close_timeout,
                )
                await conn.connect()
                return conn
            except (OSError, DqliteConnectionError, ClusterError) as exc:
                # Narrow catch: these are the transport- and cluster-level
                # failures the retry loop re-attempts. Anything wider would
                # silently log-and-re-raise programming bugs (TypeError,
                # AttributeError, …) which are better left un-instrumented
                # so the traceback points at the real source. Same pattern
                # as the _socket_looks_dead / _drain_idle narrowings.
                # OSError subsumes TimeoutError / BrokenPipeError /
                # ConnectionError / ConnectionResetError.
                logger.debug(
                    "ClusterClient.connect attempt %d/%d failed (leader=%r): %s",
                    attempt,
                    attempts_cap,
                    leader,
                    exc,
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
        return await retry_with_backoff(
            try_connect,
            max_attempts=attempts_cap,
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


def allowlist_policy(addresses: Iterable[str]) -> RedirectPolicy:
    """Build a redirect policy that accepts only the given addresses.

    Useful for the common case: "only allow redirects to hosts I've
    explicitly seed-listed." Addresses are matched by exact string
    equality — callers that need CIDR / DNS / wildcard matching should
    supply their own callable.

    Accepts any iterable (list, set, tuple, generator, dict_keys). The
    iterable is materialized into a frozen set once, so passing a
    generator is safe — the returned closure doesn't re-iterate.
    """
    allowed = frozenset(addresses)

    def policy(addr: str) -> bool:
        return addr in allowed

    return policy
