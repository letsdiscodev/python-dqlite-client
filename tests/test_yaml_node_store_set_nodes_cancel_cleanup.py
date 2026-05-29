"""Cancel-recovery and tempfile-cleanup invariants for ``YamlNodeStore.set_nodes``."""

from __future__ import annotations

import asyncio
import errno
import os
import threading
from pathlib import Path

import pytest

from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_cancel_before_lock_leaves_state_unchanged(
    tmp_path: Path,
) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    original = [NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)]
    await store.set_nodes(original)

    # Park the lock so the cancel delivers on the lock-acquire boundary.
    async with store._lock:
        task = asyncio.create_task(
            store.set_nodes([NodeInfo(node_id=2, address="127.0.0.2:9001", role=NodeRole.VOTER)])
        )
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert tuple(await store.get_nodes()) == tuple(original)
    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    assert tuple(await on_disk.get_nodes()) == tuple(original)
    orphans = list(tmp_path.glob(".nodes.yaml.*.tmp"))
    assert orphans == [], f"orphan tempfile leak after lock-cancel: {orphans}"


@pytest.mark.asyncio
async def test_set_nodes_orphan_tempfile_cleaned_up_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(errno.EINTR, "simulated interrupt")

    monkeypatch.setattr("dqliteclient.node_store.os.replace", failing_replace)

    with pytest.raises(OSError):
        await store.set_nodes([NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)])

    orphans = list(tmp_path.glob(".nodes.yaml.*.tmp"))
    assert orphans == [], (
        f"orphan tempfile leak after os.replace failure: {orphans}; "
        f"the finally: os.unlink cleanup did not run"
    )


@pytest.mark.asyncio
async def test_set_nodes_cancel_during_to_thread_leaves_no_orphan_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    started = threading.Event()
    can_finish = threading.Event()

    real_replace = os.replace
    replace_calls: list[tuple[str, str]] = []

    def slow_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        can_finish.wait(timeout=5.0)
        replace_calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_replace)

    task = asyncio.create_task(
        store.set_nodes([NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)])
    )

    # Poll in the loop (not a synchronous wait) to avoid serialising against
    # the asyncio.to_thread executor.
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker thread did not reach slow_replace within the test budget"

    # Cancel while the worker is parked; asyncio cannot interrupt the thread.
    task.cancel()
    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Filter to paths that still exist: dirent-cache shapes can return entries via
    # ``glob`` that the worker already renamed/unlinked by assert time.
    orphans = [p for p in tmp_path.glob(".nodes.yaml.*.tmp") if p.exists()]
    assert orphans == [], (
        f"orphan tempfile leak after cancel-during-to_thread: {orphans}; "
        f"replace_calls={replace_calls}; "
        f"the finally-arm cleanup or the rename did not converge to "
        f"a self-consistent on-disk state"
    )
