"""Pin: ``YamlNodeStore.create`` loads via ``asyncio.to_thread`` so other coroutines
keep running; the sync constructor's blocking ``os.read`` + ``yaml.safe_load`` would
freeze the event loop on a slow disk.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


@pytest.mark.asyncio
async def test_create_returns_populated_store(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: voter}\n")

    store = await YamlNodeStore.create(yaml_file)
    nodes = await store.get_nodes()
    assert len(nodes) == 1
    assert nodes[0].node_id == 1
    assert nodes[0].address == "h:9001"


@pytest.mark.asyncio
async def test_create_keeps_loop_responsive_during_load(tmp_path: Path) -> None:
    """A sibling task makes progress while ``create`` runs a blocking (slow-disk) load."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: voter}\n")

    progress: list[int] = []

    async def sibling() -> None:
        for i in range(5):
            await asyncio.sleep(0.005)
            progress.append(i)

    real_load = YamlNodeStore._load_from_disk

    def slow_load(self: YamlNodeStore) -> tuple[object, ...]:
        time.sleep(0.05)
        return real_load(self)

    sibling_task = asyncio.create_task(sibling())
    try:
        original = YamlNodeStore._load_from_disk
        YamlNodeStore._load_from_disk = slow_load  # type: ignore[assignment]
        try:
            store = await YamlNodeStore.create(yaml_file)
        finally:
            YamlNodeStore._load_from_disk = original
        await sibling_task
    finally:
        if not sibling_task.done():
            sibling_task.cancel()

    assert len(store._nodes) == 1
    # Without to_thread the sibling couldn't progress during the blocking sleep.
    assert len(progress) >= 3


@pytest.mark.asyncio
async def test_create_surfaces_malformed_file_as_cluster_error(tmp_path: Path) -> None:
    """The async factory keeps the constructor's fail-fast posture: malformed files
    surface at construction."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("not a list\n")

    with pytest.raises(ClusterError):
        await YamlNodeStore.create(yaml_file)


@pytest.mark.asyncio
async def test_create_set_nodes_round_trip(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: voter}\n")

    store = await YamlNodeStore.create(yaml_file)
    from dqliteclient.node_store import NodeInfo
    from dqlitewire import NodeRole

    await store.set_nodes(
        [NodeInfo(node_id=2, address="h:9002", role=NodeRole.VOTER)],
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 1
    assert nodes[0].node_id == 2
