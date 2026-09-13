"""MemoryNodeStore and NodeInfo: validation, dedup, atomicity, fork guard, generator input."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import dqliteclient
from dqliteclient.exceptions import InterfaceError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


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
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        nodes = await store.get_nodes()
        for node in nodes:
            assert node.role == 0, f"Expected role=0 (VOTER), got role={node.role}"

    async def test_initial_node_ids_start_at_one(self) -> None:
        """Node IDs start at 1; 0 means "no node" in dqlite."""
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
        assert hasattr(dqliteclient, "NodeInfo")
        assert dqliteclient.NodeInfo is NodeInfo

    async def test_get_nodes_returns_immutable_sequence(self) -> None:
        """The store hands out an immutable tuple of frozen NodeInfo."""
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

        store = MemoryNodeStore(["a:9001", "b:9001"])
        nodes = await store.get_nodes()
        assert all(isinstance(n.role, NodeRole) for n in nodes)
        assert all(n.role == NodeRole.VOTER for n in nodes)
        assert nodes[0].role == 0

    async def test_node_info_role_accepts_noderole(self) -> None:
        from dqlitewire import NodeRole

        info = NodeInfo(node_id=1, address="h:1", role=NodeRole.STANDBY)
        assert info.role is NodeRole.STANDBY
        assert info.role == 1


def test_memory_store_addresses_kwarg_name() -> None:
    """``addresses=`` is the preferred kwarg."""
    from dqliteclient import MemoryNodeStore

    store = MemoryNodeStore(addresses=["host:9001"])
    import asyncio

    nodes = asyncio.run(store.get_nodes())
    assert len(nodes) == 1
    assert nodes[0].address == "host:9001"


@pytest.mark.parametrize(
    "bad",
    [
        "no-port",
        "host:abc",
        "host:99999",
        "::1:9001",  # unbracketed IPv6
    ],
)
def test_init_rejects_malformed_address(bad: str) -> None:
    with pytest.raises(ValueError, match="not a valid"):
        MemoryNodeStore([bad])


def test_init_accepts_well_formed_addresses() -> None:
    store = MemoryNodeStore(["localhost:9001", "[::1]:9002"])
    nodes = list(store._nodes)
    assert len(nodes) == 2
    assert nodes[0].address == "localhost:9001"
    assert nodes[1].address == "[::1]:9002"


@pytest.mark.asyncio
async def test_set_nodes_rejects_malformed_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(ValueError, match="not a valid"):
        await store.set_nodes([NodeInfo(node_id=1, address="bogus:notaport", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_accepts_well_formed_addresses() -> None:
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="localhost:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="[::1]:9002", role=NodeRole.STANDBY),
        ]
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 2


@pytest.mark.asyncio
async def test_set_nodes_dedups_case_insensitive_hostname() -> None:
    """Mixed-case hostname variants are one canonical address (RFC 1035); keep one entry."""
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="Node1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="node1:9001", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="node2:9001", role=NodeRole.SPARE),
        ]
    )
    nodes = await store.get_nodes()
    # First-wins dedup: Node1 kept, the second mixed-case variant dropped.
    assert len(nodes) == 2, (
        f"expected 2 nodes after canonical dedup of Node1:9001 vs node1:9001, "
        f"got {len(nodes)}: {[n.address for n in nodes]}"
    )
    addresses = {n.address.lower() for n in nodes}
    assert "node1:9001" in addresses
    assert "node2:9001" in addresses


@pytest.mark.asyncio
async def test_set_nodes_dedups_ipv6_short_vs_long_form() -> None:
    """IPv6 short and long forms parse to the same canonical tuple; only one survives."""
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="[::1]:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="[0:0:0:0:0:0:0:1]:9001", role=NodeRole.STANDBY),
        ]
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 1, (
        f"expected 1 node after IPv6 short-vs-long-form dedup, got {len(nodes)}: "
        f"{[n.address for n in nodes]}"
    )


@pytest.mark.asyncio
async def test_memory_store_get_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryNodeStore(addresses=["h:9001"])
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.get_nodes()


@pytest.mark.asyncio
async def test_memory_store_set_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryNodeStore(addresses=["h:9001"])
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.set_nodes([NodeInfo(1, "h:9001", NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_yaml_store_get_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cluster.yaml"
    path.write_text("- Address: h:9001\n  ID: 1\n  Role: voter\n")
    store = YamlNodeStore(path)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.get_nodes()


@pytest.mark.asyncio
async def test_yaml_store_set_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cluster.yaml"
    path.write_text("- Address: h:9001\n  ID: 1\n  Role: voter\n")
    store = YamlNodeStore(path)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.set_nodes([NodeInfo(1, "h:9002", NodeRole.VOTER)])


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (NodeRole.VOTER, NodeRole.VOTER),
        (NodeRole.STANDBY, NodeRole.STANDBY),
        (NodeRole.SPARE, NodeRole.SPARE),
        (0, NodeRole.VOTER),
        (1, NodeRole.STANDBY),
        (2, NodeRole.SPARE),
    ],
)
def test_node_info_accepts_canonical_roles(role: NodeRole | int, expected: NodeRole) -> None:
    node = NodeInfo(node_id=1, address="leader:9001", role=role)  # type: ignore[arg-type]
    assert node.role == expected
    assert isinstance(node.role, NodeRole)


@pytest.mark.parametrize("bogus_role", [3, 4, 999])
def test_node_info_rejects_unknown_roles(bogus_role: int) -> None:
    with pytest.raises(ValueError, match="role"):
        NodeInfo(node_id=1, address="leader:9001", role=bogus_role)  # type: ignore[arg-type]


@pytest.fixture
def two_nodes() -> list[NodeInfo]:
    return [
        NodeInfo(node_id=1, address="host-a:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="host-b:9001", role=NodeRole.VOTER),
    ]


@pytest.fixture
def invalid_address_nodes() -> list[NodeInfo]:
    """Empty post-strip address triggers ValueError mid-iteration (duplicates are deduped,
    not rejected)."""
    return [
        NodeInfo(node_id=1, address="host-a:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="   ", role=NodeRole.VOTER),
    ]


def test_memory_store_atomic_on_invalid_address(
    two_nodes: list[NodeInfo],
    invalid_address_nodes: list[NodeInfo],
) -> None:
    async def _run() -> None:
        store = MemoryNodeStore()
        await store.set_nodes(two_nodes)
        original = await store.get_nodes()
        with pytest.raises(ValueError):
            await store.set_nodes(invalid_address_nodes)
        assert list(await store.get_nodes()) == list(original)

    asyncio.run(_run())


def test_memory_store_atomic_on_non_string_address(
    two_nodes: list[NodeInfo],
) -> None:
    async def _run() -> None:
        store = MemoryNodeStore()
        await store.set_nodes(two_nodes)
        original = await store.get_nodes()
        bad = [NodeInfo(node_id=99, address=12345, role=NodeRole.VOTER)]  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await store.set_nodes(bad)
        assert list(await store.get_nodes()) == list(original)

    asyncio.run(_run())


def test_yaml_store_atomic_on_invalid_address(
    tmp_path: Path,
    two_nodes: list[NodeInfo],
    invalid_address_nodes: list[NodeInfo],
) -> None:
    """Failed validation leaves on-disk file and in-memory state intact, no orphan tmpfile."""

    async def _run() -> None:
        path = tmp_path / "nodes.yaml"
        store = YamlNodeStore(path)
        await store.set_nodes(two_nodes)
        original = await store.get_nodes()
        with pytest.raises(ValueError):
            await store.set_nodes(invalid_address_nodes)
        assert list(await store.get_nodes()) == list(original)
        existing = sorted(p.name for p in tmp_path.iterdir())
        assert existing == ["nodes.yaml"], f"validation failure left an orphan file: {existing}"

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_set_nodes_rejects_non_string_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(TypeError, match="(?i)address must be"):
        await store.set_nodes(
            [NodeInfo(node_id=1, address=12345, role=NodeRole.VOTER)]  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_set_nodes_rejects_empty_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(ValueError, match="(?i)non-empty"):
        await store.set_nodes([NodeInfo(node_id=1, address="", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_rejects_whitespace_only_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(ValueError, match="(?i)non-empty"):
        await store.set_nodes([NodeInfo(node_id=1, address="   ", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_strips_whitespace() -> None:
    store = MemoryNodeStore()
    await store.set_nodes([NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER)])
    nodes = await store.get_nodes()
    assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_set_nodes_dedups_duplicates_first_wins() -> None:
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="  127.0.0.1:9001  ", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="127.0.0.1:9002", role=NodeRole.SPARE),
        ]
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 2
    assert nodes[0].node_id == 1
    assert nodes[1].node_id == 3


@pytest.mark.asyncio
async def test_set_nodes_rebuilds_nodeinfo_when_address_stripped() -> None:
    """A stripped address yields a new NodeInfo instance (frozen dataclass rebuild)."""
    store = MemoryNodeStore()
    original = NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER)
    await store.set_nodes([original])
    nodes = await store.get_nodes()
    stored = nodes[0]
    assert stored is not original
    assert stored.address == "127.0.0.1:9001"
    assert stored.node_id == original.node_id
    assert stored.role == original.role


def test_node_id_zero_rejected() -> None:
    with pytest.raises(ValueError, match=r"node_id must be >= 1"):
        NodeInfo(node_id=0, address="host:9000", role=NodeRole.VOTER)


def test_node_id_negative_rejected() -> None:
    with pytest.raises(ValueError, match=r"node_id must be >= 1"):
        NodeInfo(node_id=-5, address="host:9000", role=NodeRole.VOTER)


def test_node_id_bool_true_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id=True, address="host:9000", role=NodeRole.VOTER)


def test_node_id_bool_false_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id=False, address="host:9000", role=NodeRole.VOTER)


def test_node_id_float_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id=1.5, address="host:9000", role=NodeRole.VOTER)  # type: ignore[arg-type]


def test_node_id_str_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id="1", address="host:9000", role=NodeRole.VOTER)  # type: ignore[arg-type]


def test_node_id_one_accepted() -> None:
    info = NodeInfo(node_id=1, address="host:9000", role=NodeRole.VOTER)
    assert info.node_id == 1


def test_large_node_id_accepted() -> None:
    info = NodeInfo(node_id=2**40, address="host:9000", role=NodeRole.VOTER)
    assert info.node_id == 2**40


@pytest.mark.asyncio
async def test_set_nodes_accepts_generator_expression() -> None:
    store = MemoryNodeStore()
    nodes_gen = (
        NodeInfo(node_id=i + 1, address=f"127.0.0.1:{9001 + i}", role=NodeRole.VOTER)
        for i in range(3)
    )
    await store.set_nodes(nodes_gen)  # type: ignore[arg-type]
    nodes = await store.get_nodes()
    assert len(nodes) == 3
    assert [n.node_id for n in nodes] == [1, 2, 3]


@pytest.mark.asyncio
async def test_set_nodes_generator_still_subject_to_wire_cap() -> None:
    from dqlitewire import MAX_NODE_COUNT as wire_max

    store = MemoryNodeStore()
    over_cap = wire_max + 1
    nodes_gen = (
        NodeInfo(node_id=i + 1, address=f"127.0.0.1:{1000 + i}", role=NodeRole.VOTER)
        for i in range(over_cap)
    )
    with pytest.raises(ValueError, match="(?i)too many nodes"):
        await store.set_nodes(nodes_gen)  # type: ignore[arg-type]
