"""Cluster management and leader detection for dqlite."""

import asyncio
import random

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


class ClusterClient:
    """Client with automatic leader detection and failover."""

    def __init__(
        self,
        node_store: NodeStore,
        *,
        timeout: float = 10.0,
    ) -> None:
        """Initialize cluster client.

        Args:
            node_store: Store for cluster node information
            timeout: Connection timeout in seconds
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self._node_store = node_store
        self._timeout = timeout

    @classmethod
    def from_addresses(cls, addresses: list[str], timeout: float = 10.0) -> "ClusterClient":
        """Create cluster client from list of addresses."""
        store = MemoryNodeStore(addresses)
        return cls(store, timeout=timeout)

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

    async def connect(self, database: str = "default") -> DqliteConnection:
        """Connect to the cluster leader.

        Returns a connection to the current leader.
        """

        async def try_connect() -> DqliteConnection:
            leader = await self.find_leader()
            conn = DqliteConnection(leader, database=database, timeout=self._timeout)
            await conn.connect()
            return conn

        return await retry_with_backoff(
            try_connect,
            max_attempts=5,
            retryable_exceptions=(
                DqliteConnectionError,
                ClusterError,
                OperationalError,
                OSError,
                TimeoutError,
            ),
        )

    async def update_nodes(self, nodes: list[NodeInfo]) -> None:
        """Update the node store with new node information."""
        await self._node_store.set_nodes(nodes)
