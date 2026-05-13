"""Cancel-recovery and tempfile-cleanup invariants for
``YamlNodeStore.set_nodes``.

Two surfaces, both load-bearing for the membership-write contract
under cancel / failure:

1. **Cancel landing on the lock-acquire await.** The pre-lock cancel
   surface always existed (the only ``await`` is the ``async with
   self._lock`` until the to_thread refactor opened the second
   surface). State (in-memory + on-disk) must not change.

2. **Cancel / failure mid-write under the to_thread refactor.** Now
   that the body runs on a worker thread (companion finding
   ``yaml-store-set-nodes-blocks-event-loop-during-fsync-and-rename.md``),
   ``os.replace`` failing mid-write must leave the parent directory
   pristine — no orphan ``.nodes.yaml.RANDOM.tmp`` files. Repeated
   cancel-storms / SIGTERM in a retry loop would otherwise hit the
   filesystem inode limit.
"""

from __future__ import annotations

import asyncio
import errno
import os
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

    # Park the lock so the next set_nodes hits the await and cancel
    # delivers on the lock-acquire boundary.
    async with store._lock:
        task = asyncio.create_task(
            store.set_nodes([NodeInfo(node_id=2, address="127.0.0.2:9001", role=NodeRole.VOTER)])
        )
        # Yield enough times so the task is parked on the lock.
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # State unchanged — neither in-memory nor on disk.
    assert tuple(await store.get_nodes()) == tuple(original)
    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    assert tuple(await on_disk.get_nodes()) == tuple(original)
    # No orphan tempfile.
    orphans = list(tmp_path.glob(".nodes.yaml.*.tmp"))
    assert orphans == [], f"orphan tempfile leak after lock-cancel: {orphans}"


@pytest.mark.asyncio
async def test_set_nodes_orphan_tempfile_cleaned_up_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os.replace`` failure (simulating EINTR / fs error mid-rename)
    must trigger the ``finally`` ``os.unlink`` cleanup so no orphan
    tempfile lingers. Forward-compat invariant for the cancel-during-
    to_thread surface."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(errno.EINTR, "simulated interrupt")

    # Patch the module-level reference used in _write_atomic_sync.
    monkeypatch.setattr("dqliteclient.node_store.os.replace", failing_replace)

    with pytest.raises(OSError):
        await store.set_nodes([NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)])

    # No orphan tempfile — the finally arm ran the unlink.
    orphans = list(tmp_path.glob(".nodes.yaml.*.tmp"))
    assert orphans == [], (
        f"orphan tempfile leak after os.replace failure: {orphans}; "
        f"the finally: os.unlink cleanup did not run"
    )
