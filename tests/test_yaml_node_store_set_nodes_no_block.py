"""``set_nodes`` dispatches the atomic-rename ritual to a worker thread so a slow
fsync does not freeze the event loop."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_yields_during_slow_fsync(tmp_path: Path) -> None:
    """Block fsync ~500 ms; a concurrent ticker must still tick during set_nodes."""
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)
    new_nodes = [NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)]

    real_fsync = os.fsync

    def slow_fsync(fd: int) -> None:
        time.sleep(0.5)
        real_fsync(fd)

    tick_times: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.02)
            tick_times.append(time.monotonic())

    with patch("dqliteclient.node_store.os.fsync", new=slow_fsync):
        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await store.set_nodes(new_nodes)
        done = time.monotonic()
        stop.set()
        await ticker_task

    ticks_during_write = [t for t in tick_times if started < t < done]
    assert len(ticks_during_write) >= 3, (
        f"only {len(ticks_during_write)} ticks landed during the "
        f"{done - started:.3f}s set_nodes call window; the event loop "
        f"was blocked. Expected the loop to remain responsive via "
        f"asyncio.to_thread dispatch."
    )


@pytest.mark.asyncio
async def test_set_nodes_still_writes_payload_atomically(tmp_path: Path) -> None:
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)
    new_nodes = [
        NodeInfo(node_id=1, address="10.0.0.1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="10.0.0.2:9001", role=NodeRole.SPARE),
    ]
    await store.set_nodes(new_nodes)
    assert store_path.exists()
    text = store_path.read_text(encoding="utf-8")
    assert "10.0.0.1:9001" in text
    assert "10.0.0.2:9001" in text
    assert {n.address for n in await store.get_nodes()} == {
        "10.0.0.1:9001",
        "10.0.0.2:9001",
    }
