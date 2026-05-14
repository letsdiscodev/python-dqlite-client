"""Pin: ``YamlNodeStore.set_nodes`` does not leave disk-vs-memory
divergent when a cancel lands between the worker-thread return and
the in-memory assignment.

The body dispatches the atomic write to ``asyncio.to_thread``. asyncio
cannot abort the worker thread mid-flight; the cancel parks until
the future resolves and then re-raises ``CancelledError`` at the
await boundary. Pre-fix, the assignment ``self._nodes = normalised``
sat AFTER the await — a cancel landing on that boundary would leave
the on-disk file with the NEW membership and in-memory ``_nodes``
with the OLD. The divergence is permanent for the process lifetime
and silently breaks membership-change → re-enumerate flows.

The fix shields the await + assignment together so the cancel
re-raises only after both have committed.
"""

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
    """Park the worker thread mid-write; cancel the awaiting task;
    release the worker so the rename completes. After the cancel
    surfaces, in-memory ``_nodes`` MUST match what landed on disk —
    no divergent state."""
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

    # Deliver the cancel WHILE the worker is parked.
    task.cancel()
    # Release the worker so the rename completes; the to_thread
    # future resolves and the cancel re-raises at the await boundary.
    can_finish.set()

    # The cancel surfaces — but the shield ensures the in-memory
    # assignment ran before the cancel re-raised.
    with pytest.raises(asyncio.CancelledError):
        await task

    # Wait for the worker thread to actually finish the rename so
    # the disk-state check below sees the committed result.
    for _ in range(200):
        if finished.is_set():
            break
        await asyncio.sleep(0.01)
    assert finished.is_set(), "worker thread did not finish slow_replace"

    # On-disk state: the rename completed, so the file holds the
    # NEW membership.
    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    disk_nodes = tuple(await on_disk.get_nodes())
    assert disk_nodes == tuple(new_nodes), (
        f"disk should hold the NEW membership after the rename completed; got {disk_nodes}"
    )

    # In-memory state: must match what's on disk. Pre-fix this was
    # the OLD (empty) membership — divergent and silently broken.
    in_memory = tuple(await store.get_nodes())
    assert in_memory == disk_nodes, (
        f"in-memory _nodes diverged from on-disk after cancel: "
        f"in_memory={in_memory}, on_disk={disk_nodes}. The cancel "
        f"re-raised between the to_thread return and the assignment, "
        f"leaving permanent divergence."
    )
