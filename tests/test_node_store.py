"""Tests for node store."""

import dqliteclient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo


class TestMemoryNodeStore:
    async def test_empty_store(self) -> None:
        store = MemoryNodeStore()
        nodes = await store.get_nodes()
        assert nodes == []

    async def test_initial_addresses(self) -> None:
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        nodes = await store.get_nodes()
        assert len(nodes) == 2
        assert nodes[0].address == "localhost:9001"
        assert nodes[1].address == "localhost:9002"

    async def test_initial_nodes_have_voter_role(self) -> None:
        """Initial nodes should be VOTER (role=0), not STANDBY."""
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        nodes = await store.get_nodes()
        for node in nodes:
            assert node.role == 0, f"Expected role=0 (VOTER), got role={node.role}"

    async def test_initial_node_ids_start_at_one(self) -> None:
        """Node IDs should start at 1, since 0 means 'no node' in dqlite."""
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        nodes = await store.get_nodes()
        assert nodes[0].node_id == 1
        assert nodes[1].node_id == 2

    async def test_set_nodes(self) -> None:
        store = MemoryNodeStore()
        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=1),
            NodeInfo(node_id=2, address="node2:9002", role=2),
        ]
        await store.set_nodes(nodes)

        result = await store.get_nodes()
        assert len(result) == 2
        assert result[0].node_id == 1
        assert result[1].address == "node2:9002"

    def test_nodeinfo_exported_from_package(self) -> None:
        """NodeInfo should be importable from the top-level package."""
        assert hasattr(dqliteclient, "NodeInfo")
        assert dqliteclient.NodeInfo is NodeInfo

    async def test_get_nodes_returns_copy(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        nodes1 = await store.get_nodes()
        nodes2 = await store.get_nodes()
        assert nodes1 is not nodes2
