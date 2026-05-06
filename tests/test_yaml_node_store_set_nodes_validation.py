"""Pin: ``YamlNodeStore.set_nodes`` enforces the same
strip / dedup / ``_parse_address`` validation pipeline as
``MemoryNodeStore.set_nodes``.

Without this, a process calling ``cluster_info() -> set_nodes(...)``
could persist whitespace-laden / duplicated / malformed entries to
disk, and the same ``_load_from_disk`` loader would later refuse the
file the writer just wrote — postcondition violation (``set_nodes``
succeeded but the next process startup raises ``ValueError`` on the
very bytes ``set_nodes`` wrote).

Mirrors the existing
``test_node_store_set_nodes_validation.py`` MemoryNodeStore pins
on the persistent store side.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


def _make_store(path: Path) -> YamlNodeStore:
    """Construct a store rooted at ``path``. The constructor reads
    from disk if the file exists; otherwise the in-memory tuple
    starts empty."""
    return YamlNodeStore(path)


@pytest.mark.asyncio
async def test_set_nodes_rejects_non_string_address() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(TypeError, match="(?i)address must be"):
            await store.set_nodes(
                [NodeInfo(node_id=1, address=12345, role=NodeRole.VOTER)]  # type: ignore[arg-type]
            )


@pytest.mark.asyncio
async def test_set_nodes_rejects_empty_address() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(ValueError, match="(?i)non-empty"):
            await store.set_nodes([NodeInfo(node_id=1, address="", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_rejects_whitespace_only_address() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(ValueError, match="(?i)non-empty"):
            await store.set_nodes([NodeInfo(node_id=1, address="   ", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_rejects_malformed_host_port() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        with pytest.raises(ValueError, match="host:port"):
            await store.set_nodes(
                [NodeInfo(node_id=1, address="not_a_valid_address", role=NodeRole.VOTER)]
            )


@pytest.mark.asyncio
async def test_set_nodes_strips_whitespace() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
        await store.set_nodes(
            [NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER)]
        )
        nodes = await store.get_nodes()
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_set_nodes_dedups_duplicates_first_wins() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")
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
async def test_set_nodes_persists_canonical_form_to_disk() -> None:
    """End-to-end pin: a strip/dedup-required call to ``set_nodes``
    writes the CANONICAL form to disk, and a fresh ``YamlNodeStore``
    instance loads cleanly without ``_load_from_disk``'s validators
    rejecting the bytes the writer just wrote."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        store = _make_store(path)
        await store.set_nodes(
            [
                NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER),
                NodeInfo(node_id=2, address="127.0.0.1:9001", role=NodeRole.STANDBY),
            ]
        )
        # Reload from disk via a fresh store; would-have-failed before
        # the validation parity fix because the YAML payload contained
        # the un-stripped, duplicated entries.
        reloaded = YamlNodeStore(path)
        nodes = await reloaded.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_concurrent_set_nodes_serialised_under_lock() -> None:
    """Two concurrent ``set_nodes`` calls must serialise through the
    instance lock — no validation-pre-pass races, no torn writes."""
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")

        async def writer(addr: str) -> None:
            await store.set_nodes([NodeInfo(node_id=1, address=addr, role=NodeRole.VOTER)])

        # Race ten writers; the last one's value is the persisted state.
        await asyncio.gather(*(writer(f"127.0.0.{i}:9001") for i in range(1, 11)))
        nodes = await store.get_nodes()
        assert len(nodes) == 1
