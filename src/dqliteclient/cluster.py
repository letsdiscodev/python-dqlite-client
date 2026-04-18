"""Cluster management and leader detection for dqlite."""

import asyncio
import random
from collections.abc import Callable, Iterable

from dqliteclient.connection import DqliteConnection, _parse_address
from dqliteclient.exceptions import (
    ClusterError,
    DqliteConnectionError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, NodeStore
from dqliteclient.protocol import DqliteProtocol
from dqliteclient.retry import retry_with_backoff

# Type alias for a redirect-target policy. Returns True if the address
# should be accepted, False to reject with a ClusterError.
RedirectPolicy = Callable[[str], bool]


class ClusterClient:
    """Client with automatic leader detection and failover."""

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
                callable or the ``only_nodes_in_store`` helper to
                constrain where redirects can go.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self._node_store = node_store
        self._timeout = timeout
        self._redirect_policy = redirect_policy

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
            raise ClusterError(f"Leader redirect to {address!r} rejected by redirect_policy")

    async def find_leader(self) -> str:
        """Find the current cluster leader.

        Returns the leader address.
        """
        nodes = await self._node_store.get_nodes()

        if not nodes:
            raise ClusterError("No nodes configured")

        # Shuffle first so repeated callers don't stampede the same node;
        # then stable-sort by role so voters come before non-voters.
        # Standby/spare nodes can never become leader (their LEADER
        # response is always (0, "")), so probing them first wastes RTTs.
        nodes = list(nodes)
        random.shuffle(nodes)
        nodes.sort(key=lambda n: 0 if n.role == 0 else 1)

        errors: list[str] = []
        last_exc: BaseException | None = None

        for node in nodes:
            try:
                leader_address = await asyncio.wait_for(
                    self._query_leader(node.address), timeout=self._timeout
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
                errors.append(f"{node.address}: timed out")
                last_exc = e
                continue
            except (DqliteConnectionError, ProtocolError, OperationalError, OSError) as e:
                # Narrow the catch so programming bugs (TypeError, KeyError,
                # etc.) propagate directly instead of being stringified into
                # a retryable ClusterError.
                errors.append(f"{node.address}: {e}")
                last_exc = e
                continue

        raise ClusterError(f"Could not find leader. Errors: {'; '.join(errors)}") from last_exc

    async def _query_leader(self, address: str) -> str | None:
        """Query a node for the current leader."""
        host, port = _parse_address(address)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._timeout,
            )
        except (TimeoutError, OSError):
            return None

        try:
            protocol = DqliteProtocol(reader, writer, timeout=self._timeout)
            await protocol.handshake()
            node_id, leader_addr = await protocol.get_leader()

            if not leader_addr and node_id != 0:
                # Non-zero node_id with empty address: this node is the leader
                return address
            elif leader_addr:
                # Non-empty address: redirect to the reported leader
                return leader_addr
            else:
                # node_id=0 and empty address: no leader known
                return None
        finally:
            writer.close()

    async def connect(
        self,
        database: str = "default",
        *,
        max_total_rows: int | None = 10_000_000,
    ) -> DqliteConnection:
        """Connect to the cluster leader.

        Returns a connection to the current leader. ``max_total_rows``
        is forwarded to the underlying :class:`DqliteConnection` so
        callers (including :class:`ConnectionPool`) can tune the
        cumulative row cap from one place.
        """

        async def try_connect() -> DqliteConnection:
            leader = await self.find_leader()
            conn = DqliteConnection(
                leader,
                database=database,
                timeout=self._timeout,
                max_total_rows=max_total_rows,
            )
            await conn.connect()
            return conn

        # Retry only transport-level errors. Leader-change OperationalError
        # codes are now reclassified into DqliteConnectionError at connect
        # time (see #148), so we no longer need OperationalError in the
        # retry set — that avoids amplifying a schema/SQL error into 5 ×
        # N_nodes RTTs before propagating.
        return await retry_with_backoff(
            try_connect,
            max_attempts=3,
            retryable_exceptions=(
                DqliteConnectionError,
                ClusterError,
                OSError,
                TimeoutError,
            ),
        )

    async def update_nodes(self, nodes: list[NodeInfo]) -> None:
        """Update the node store with new node information."""
        await self._node_store.set_nodes(nodes)


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
