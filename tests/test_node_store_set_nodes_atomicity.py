"""``set_nodes`` validation failure leaves the prior node list intact (atomicity)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


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
