"""Cluster management and leader detection for dqlite."""

import asyncio

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import ClusterError
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
        self._node_store = node_store
        self._timeout = timeout
        self._leader_address: str | None = None

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

        errors: list[str] = []

        for node in nodes:
            try:
                leader_address = await self._query_leader(node.address)
                if leader_address:
                    self._leader_address = leader_address
                    return leader_address
            except Exception as e:
                errors.append(f"{node.address}: {e}")
                continue

        raise ClusterError(f"Could not find leader. Errors: {'; '.join(errors)}")

    async def _query_leader(self, address: str) -> str | None:
        """Query a node for the current leader."""
        host, port_str = address.rsplit(":", 1)
        port = int(port_str)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._timeout,
            )
        except (TimeoutError, OSError):
            return None

        protocol = DqliteProtocol(reader, writer)

        try:
            await protocol.handshake()
            node_id, leader_addr = await protocol.get_leader()

            # If address is empty, this node is the leader
            if not leader_addr:
                return address

            return leader_addr
        finally:
            protocol.close()
            await protocol.wait_closed()

    async def connect(self, database: str = "default") -> DqliteConnection:
        """Connect to the cluster leader.

        Returns a connection to the current leader.
        """

        async def try_connect() -> DqliteConnection:
            leader = await self.find_leader()
            conn = DqliteConnection(leader, database=database, timeout=self._timeout)
            await conn.connect()
            return conn

        return await retry_with_backoff(try_connect, max_attempts=5)

    async def update_nodes(self, nodes: list[NodeInfo]) -> None:
        """Update the node store with new node information."""
        await self._node_store.set_nodes(nodes)
