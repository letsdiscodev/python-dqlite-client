"""``set_nodes`` holds the lock for the full duration of the shielded inner write, so a
cancelled writer's orphaned shielded commit cannot race a concurrent writer that would
otherwise acquire the lock freed by ``async with``'s ``__aexit__``."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_cancel_releases_lock_lets_second_writer_race_orphan_shielded_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer A is cancelled mid-write; the lock must stay held until its shielded
    inner write completes, so a concurrent writer B parks on the lock rather than
    racing the orphaned commit."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    nodes_a = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]
    nodes_b = [
        NodeInfo(node_id=3, address="127.0.0.3:9001", role=NodeRole.VOTER),
    ]

    # Park writer A's worker inside ``os.replace`` so we can cancel it mid-write.
    a_in_replace = threading.Event()
    a_can_finish = threading.Event()
    real_replace = os.replace
    parked_once = threading.Event()

    def parking_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        # Park only the first caller (writer A); later calls pass through.
        if not parked_once.is_set():
            parked_once.set()
            a_in_replace.set()
            a_can_finish.wait(timeout=5.0)
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", parking_replace)

    task_a = asyncio.create_task(store.set_nodes(nodes_a))

    for _ in range(200):
        if a_in_replace.is_set():
            break
        await asyncio.sleep(0.01)
    assert a_in_replace.is_set(), "writer A did not enter parking_replace"

    task_a.cancel()

    # Yield so the cancel can propagate.
    for _ in range(5):
        await asyncio.sleep(0)

    # The lock is still held even though task A has unwound, because the shield
    # wraps the entire locked region; a concurrent set_nodes parks on the lock.
    assert store._lock.locked(), (
        "after cancel of writer A, the store lock must still be held "
        "until the shielded inner write completes; pre-fix the lock "
        "was released by the surrounding ``async with``'s ``__aexit__`` "
        "while the shielded ``_write_and_publish`` continued in the "
        "background, opening a race for a concurrent second writer."
    )

    a_can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task_a

    for _ in range(200):
        if not store._lock.locked():
            break
        await asyncio.sleep(0.01)
    assert not store._lock.locked(), (
        "the lock must be released once the shielded inner write completes"
    )

    # A follow-on writer completes normally.
    await store.set_nodes(nodes_b)
    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    disk_nodes = tuple(await on_disk.get_nodes())
    in_memory = tuple(await store.get_nodes())
    assert disk_nodes == tuple(nodes_b)
    assert in_memory == disk_nodes
