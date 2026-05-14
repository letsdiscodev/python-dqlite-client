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


@pytest.mark.asyncio
async def test_set_nodes_cancel_during_to_thread_leaves_no_orphan_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel delivered to the ``set_nodes`` task while the worker
    thread is blocked mid-write must NOT leak an orphan tempfile.

    asyncio does not abort the worker thread on cancel; the cancel
    parks until the worker returns, then re-raises ``CancelledError``
    from the awaited ``to_thread`` future. The cleanup contract is:
    whatever state the worker leaves behind on disk must be
    self-consistent — either the rename completed (in which case the
    tempfile is gone) or the finally-arm unlink ran (in which case
    the tempfile is gone). No orphan tempfiles either way.
    """
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    started = threading.Event()
    can_finish = threading.Event()

    real_replace = os.replace
    replace_calls: list[tuple[str, str]] = []

    def slow_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        # Signal that the worker is mid-write, then park until the
        # test releases us. The cancel arrives at the awaiting
        # coroutine while this thread is blocked here.
        started.set()
        can_finish.wait(timeout=5.0)
        replace_calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_replace)

    task = asyncio.create_task(
        store.set_nodes([NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER)])
    )

    # Wait until the worker thread is parked inside slow_replace.
    # Poll the threading.Event in the loop rather than blocking the
    # event loop on a synchronous wait — and avoid second-executor-
    # thread serialisation races with asyncio.to_thread.
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker thread did not reach slow_replace within the test budget"

    # Deliver the cancel WHILE the worker is parked. asyncio cannot
    # interrupt the thread; the cancel hangs until we release the
    # worker.
    task.cancel()

    # Release the worker so the rename completes and the to_thread
    # future resolves; the cancel then surfaces at the await.
    can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Filter glob to only paths that actually exist at assert time —
    # some filesystem/dirent-cache shapes return paths via ``glob`` that
    # have already been renamed/unlinked by the worker thread by the
    # time the assert runs. The contract is "no orphan tempfile lives
    # on disk after the operation," not "glob saw no entry at any
    # point during the race."
    orphans = [p for p in tmp_path.glob(".nodes.yaml.*.tmp") if p.exists()]
    assert orphans == [], (
        f"orphan tempfile leak after cancel-during-to_thread: {orphans}; "
        f"replace_calls={replace_calls}; "
        f"the finally-arm cleanup or the rename did not converge to "
        f"a self-consistent on-disk state"
    )
