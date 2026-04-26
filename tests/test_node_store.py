"""Tests for node store."""

import pytest

import dqliteclient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire.constants import NodeRole


class TestMemoryNodeStore:
    async def test_empty_store(self) -> None:
        store = MemoryNodeStore()
        nodes = await store.get_nodes()
        assert len(nodes) == 0

    async def test_initial_addresses(self) -> None:
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        nodes = await store.get_nodes()
        assert len(nodes) == 2
        assert nodes[0].address == "localhost:9001"
        assert nodes[1].address == "localhost:9002"

    async def test_initial_nodes_have_voter_role(self) -> None:
        """Initial nodes should be VOTER (role=NodeRole.VOTER), not STANDBY."""
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
            NodeInfo(node_id=1, address="node1:9001", role=NodeRole.STANDBY),
            NodeInfo(node_id=2, address="node2:9002", role=NodeRole.SPARE),
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

    async def test_get_nodes_returns_immutable_sequence(self) -> None:
        """The store hands out an immutable tuple of frozen NodeInfo.

        A caller therefore cannot corrupt store state by mutating the
        returned sequence or its elements.
        """
        import dataclasses

        store = MemoryNodeStore(["localhost:9001"])
        nodes = await store.get_nodes()
        assert isinstance(nodes, tuple)
        with pytest.raises(dataclasses.FrozenInstanceError):
            nodes[0].address = "evil"

    async def test_node_info_is_frozen(self) -> None:
        import dataclasses

        info = NodeInfo(node_id=1, address="h:1", role=NodeRole.VOTER)
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.address = "other"  # type: ignore[misc]

    async def test_node_info_is_hashable(self) -> None:
        info1 = NodeInfo(node_id=1, address="h:1", role=NodeRole.VOTER)
        info2 = NodeInfo(node_id=1, address="h:1", role=NodeRole.VOTER)
        assert hash(info1) == hash(info2)
        assert {info1, info2} == {info1}

    async def test_memory_store_seeds_with_noderole_voter(self) -> None:
        from dqlitewire import NodeRole

        store = MemoryNodeStore(initial_addresses=["a:9001", "b:9001"])
        nodes = await store.get_nodes()
        assert all(isinstance(n.role, NodeRole) for n in nodes)
        assert all(n.role == NodeRole.VOTER for n in nodes)
        # IntEnum comparison with raw int still works.
        assert nodes[0].role == 0

    async def test_node_info_role_accepts_noderole(self) -> None:
        from dqlitewire import NodeRole

        info = NodeInfo(node_id=1, address="h:1", role=NodeRole.STANDBY)
        assert info.role is NodeRole.STANDBY
        assert info.role == 1


def test_memory_store_addresses_kwarg_name() -> None:
    """``addresses=`` is the preferred kwarg — matches sibling APIs."""
    from dqliteclient import MemoryNodeStore

    store = MemoryNodeStore(addresses=["host:9001"])
    import asyncio

    nodes = asyncio.run(store.get_nodes())
    assert len(nodes) == 1
    assert nodes[0].address == "host:9001"


def test_memory_store_initial_addresses_still_works() -> None:
    """Legacy ``initial_addresses=`` kwarg still seeds the store."""
    from dqliteclient import MemoryNodeStore

    store = MemoryNodeStore(initial_addresses=["host:9001"])
    import asyncio

    nodes = asyncio.run(store.get_nodes())
    assert len(nodes) == 1


def test_memory_store_rejects_both_kwargs() -> None:
    import pytest

    from dqliteclient import MemoryNodeStore

    with pytest.raises(TypeError, match="Pass only one"):
        MemoryNodeStore(addresses=["a:1"], initial_addresses=["b:2"])
