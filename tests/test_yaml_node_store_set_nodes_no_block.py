"""``YamlNodeStore.set_nodes`` dispatches the sync atomic-rename ritual
to a worker thread via ``asyncio.to_thread`` so a slow fsync (NAS /
encrypted volume / SSD GC pause) does not freeze the event loop for
the duration of the disk barrier.

Pre-fix the body ran ``yaml.safe_dump``, tempfile write, ``os.fsync``,
``os.replace``, directory fsync, and the unlink cleanup synchronously
between ``async with self._lock:`` and the implicit exit — zero
await points — monopolising the loop. RPC timeouts could not fire,
heartbeat windows expired silently, and pool acquirers parked on
``_pool.get`` could not wake.
"""

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
    """Force ``os.fsync`` to block ~500 ms. A concurrent ticker must
    record ticks DURING the set_nodes call window — without the
    to_thread dispatch, the event loop is monopolised end-to-end and
    no ticks land between set_nodes start and set_nodes completion."""
    store_path = tmp_path / "nodes.yaml"
    store = YamlNodeStore(store_path)
    new_nodes = [NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)]

    real_fsync = os.fsync

    def slow_fsync(fd: int) -> None:
        # Block the worker thread for ~500 ms; the loop thread must
        # remain free so a concurrent ticker can record ticks during
        # this window.
        time.sleep(0.5)
        real_fsync(fd)

    # Record the wall-clock time of every tick. A tick that lands
    # between set_nodes_started and set_nodes_done proves the loop
    # was not blocked.
    tick_times: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.02)
            tick_times.append(time.monotonic())

    with patch("dqliteclient.node_store.os.fsync", new=slow_fsync):
        ticker_task = asyncio.create_task(ticker())
        # Give the ticker one cycle to start.
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await store.set_nodes(new_nodes)
        done = time.monotonic()
        stop.set()
        await ticker_task

    ticks_during_write = [t for t in tick_times if started < t < done]
    # The write window is bounded by 2 * 500 ms fsync = ~1 s. A 20 ms
    # ticker should land at least ~10 ticks inside that window if the
    # loop is free.
    assert len(ticks_during_write) >= 3, (
        f"only {len(ticks_during_write)} ticks landed during the "
        f"{done - started:.3f}s set_nodes call window; the event loop "
        f"was blocked. Expected the loop to remain responsive via "
        f"asyncio.to_thread dispatch."
    )


@pytest.mark.asyncio
async def test_set_nodes_still_writes_payload_atomically(tmp_path: Path) -> None:
    """Smoke test: after the to_thread dispatch the file must still
    contain the expected payload (no regression of the atomic-rename
    discipline)."""
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
    # The validated tuple is also stored in memory.
    assert {n.address for n in await store.get_nodes()} == {
        "10.0.0.1:9001",
        "10.0.0.2:9001",
    }
