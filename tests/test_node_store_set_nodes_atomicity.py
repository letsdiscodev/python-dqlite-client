"""Pin: ``MemoryNodeStore.set_nodes`` and ``YamlNodeStore.set_nodes``
leave the prior node list intact when validation fails.

Go's ``TestDefaultNodeStore`` (``store_test.go:50-62``) pins this
contract for the SQL-backed default store: setting two NodeInfos
with the same address fails AND the prior contents are preserved.
The Python implementations validate pre-mutation
(``_validate_and_normalise_nodes`` raises before the
``self._nodes = ...`` assignment), so the contract is already
satisfied today — these tests pin the contract durably so a future
refactor that mutates state pre-validation breaks loudly.
"""

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
    """A second-element with an empty (post-strip) address triggers
    ``_validate_and_normalise_nodes``'s ``ValueError`` mid-iteration.
    Note: duplicate addresses are silently deduplicated rather than
    rejected — that's a Python-side design choice distinct from
    Go's ``UNIQUE constraint`` semantics."""
    return [
        NodeInfo(node_id=1, address="host-a:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="   ", role=NodeRole.VOTER),  # whitespace-only
    ]


def test_memory_store_atomic_on_invalid_address(
    two_nodes: list[NodeInfo],
    invalid_address_nodes: list[NodeInfo],
) -> None:
    """A failed ``set_nodes`` (validation raises mid-iteration) must
    leave the prior list intact — atomicity contract."""

    async def _run() -> None:
        store = MemoryNodeStore()
        await store.set_nodes(two_nodes)
        original = await store.get_nodes()
        with pytest.raises(ValueError):
            await store.set_nodes(invalid_address_nodes)
        # Atomicity: prior contents preserved.
        assert list(await store.get_nodes()) == list(original)

    asyncio.run(_run())


def test_memory_store_atomic_on_non_string_address(
    two_nodes: list[NodeInfo],
) -> None:
    """A non-string address (TypeError) must also leave prior list."""

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
    """YamlNodeStore: a failed validation must leave the on-disk file
    AND the in-memory state intact. Also: no orphan tmpfile lingers
    alongside the canonical file."""

    async def _run() -> None:
        path = tmp_path / "nodes.yaml"
        store = YamlNodeStore(path)
        await store.set_nodes(two_nodes)
        original = await store.get_nodes()
        with pytest.raises(ValueError):
            await store.set_nodes(invalid_address_nodes)
        # Atomicity (in-memory):
        assert list(await store.get_nodes()) == list(original)
        # No orphan tmpfile lingered alongside the canonical file.
        existing = sorted(p.name for p in tmp_path.iterdir())
        assert existing == ["nodes.yaml"], f"validation failure left an orphan file: {existing}"

    asyncio.run(_run())
