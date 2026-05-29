"""``set_nodes`` shields the to_thread await + in-memory assignment together, so a
cancel on that boundary cannot leave disk holding NEW membership while ``_nodes``
holds OLD."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_cancel_between_to_thread_return_and_assignment_does_not_diverge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Park the worker mid-write, cancel, release; after the cancel surfaces
    in-memory ``_nodes`` must match what landed on disk."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="127.0.0.2:9001", role=NodeRole.VOTER),
    ]

    started = threading.Event()
    can_finish = threading.Event()
    finished = threading.Event()
    real_replace = os.replace

    def slow_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        can_finish.wait(timeout=5.0)
        try:
            real_replace(src, dst)
        finally:
            finished.set()

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_replace)

    task = asyncio.create_task(store.set_nodes(new_nodes))

    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker thread did not enter slow_replace"

    # Deliver the cancel WHILE the worker is parked, then release it.
    task.cancel()
    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Wait for the worker to finish the rename so the disk check sees the commit.
    for _ in range(200):
        if finished.is_set():
            break
        await asyncio.sleep(0.01)
    assert finished.is_set(), "worker thread did not finish slow_replace"

    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    disk_nodes = tuple(await on_disk.get_nodes())
    assert disk_nodes == tuple(new_nodes), (
        f"disk should hold the NEW membership after the rename completed; got {disk_nodes}"
    )

    in_memory = tuple(await store.get_nodes())
    assert in_memory == disk_nodes, (
        f"in-memory _nodes diverged from on-disk after cancel: "
        f"in_memory={in_memory}, on_disk={disk_nodes}. The cancel "
        f"re-raised between the to_thread return and the assignment, "
        f"leaving permanent divergence."
    )
