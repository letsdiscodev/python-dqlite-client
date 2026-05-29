"""``YamlNodeStore.set_nodes`` enforces the same strip/dedup/``_parse_address``
pipeline as the loader, so it never writes bytes ``_load_from_disk`` would reject."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


def _make_store(path: Path) -> YamlNodeStore:
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
    """set_nodes writes canonical form; a fresh store reloads it without rejection."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        store = _make_store(path)
        await store.set_nodes(
            [
                NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER),
                NodeInfo(node_id=2, address="127.0.0.1:9001", role=NodeRole.STANDBY),
            ]
        )
        reloaded = YamlNodeStore(path)
        nodes = await reloaded.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_load_from_disk_strips_and_dedups_via_helper() -> None:
    """``_load_from_disk`` canonicalises hand-edited files via the same pipeline."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        path.write_text(
            "- ID: 1\n"
            "  Address: '  127.0.0.1:9001  '\n"
            "  Role: 0\n"
            "- ID: 2\n"
            "  Address: '127.0.0.1:9001'\n"
            "  Role: 0\n"
        )
        store = YamlNodeStore(path)
        nodes = await store.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_load_from_disk_rejects_invalid_address_via_helper() -> None:
    """A malformed ``host:port`` in a YAML file fails to load via the shared validator."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nodes.yaml"
        path.write_text("- ID: 1\n  Address: 'not_a_valid_address'\n  Role: 0\n")
        from dqliteclient.exceptions import ClusterError

        with pytest.raises(ClusterError, match="host:port"):
            YamlNodeStore(path)


@pytest.mark.asyncio
async def test_concurrent_set_nodes_serialised_under_lock() -> None:
    """Concurrent ``set_nodes`` calls serialise through the lock — no torn writes."""
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(Path(d) / "nodes.yaml")

        async def writer(addr: str) -> None:
            await store.set_nodes([NodeInfo(node_id=1, address=addr, role=NodeRole.VOTER)])

        await asyncio.gather(*(writer(f"127.0.0.{i}:9001") for i in range(1, 11)))
        nodes = await store.get_nodes()
        assert len(nodes) == 1
