"""Tests for node store."""

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

    async def test_get_nodes_returns_copy(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        nodes1 = await store.get_nodes()
        nodes2 = await store.get_nodes()
        assert nodes1 is not nodes2
