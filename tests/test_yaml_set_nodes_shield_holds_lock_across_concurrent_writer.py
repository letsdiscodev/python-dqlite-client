"""Pin: ``YamlNodeStore.set_nodes`` holds the lock across the
shielded inner write so a concurrent writer cannot interleave with
the shielded-but-orphaned commit of a cancelled writer.

Pre-fix the shield wrapped the inner ``_write_and_publish`` but was
INSIDE the ``async with self._lock:`` block. ``asyncio.shield`` only
shields its argument coroutine — on outer cancel, the surrounding
``async with`` ``__aexit__`` releases the lock while the shielded
write continues in the background. A concurrent ``set_nodes`` then
acquires the freed lock and races the orphaned writer on disk and
in-memory state, allowing a writer-B-then-writer-A-orphan
interleaving whose final disk state differs from the in-memory
``_nodes`` — exactly the divergence the prior shield-only fix was
supposed to prevent.

The fix holds the lock for the full duration of the shielded inner
write so concurrent writers serialise correctly even under cancel.
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
async def test_cancel_releases_lock_lets_second_writer_race_orphan_shielded_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent ``set_nodes`` callers; the first is cancelled
    mid-write. Pre-fix the cancel releases the lock while the
    shielded inner write continues in the background. The second
    writer immediately acquires the freed lock and races the
    orphaned shielded write — both can commit, with the orphan's
    disk + memory writes interleaving past the second writer's
    "successful" return.

    The pin: the lock MUST be held until the shielded inner write
    fully completes. Concretely, while writer A is parked mid-
    write and after the outer cancel has been delivered, the store
    lock must still be reported as held (so writer B parks on
    ``self._lock.acquire()``, not on its own ``to_thread`` future).
    """
    store = YamlNodeStore(tmp_path / "nodes.yaml")

    nodes_a = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]
    nodes_b = [
        NodeInfo(node_id=3, address="127.0.0.3:9001", role=NodeRole.VOTER),
    ]

    # Park writer A's worker inside ``os.replace`` so we can deliver
    # a cancel while it is mid-write.
    a_in_replace = threading.Event()
    a_can_finish = threading.Event()
    real_replace = os.replace
    parked_once = threading.Event()

    def parking_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        # Park only the first caller (writer A); every subsequent
        # call passes through directly.
        if not parked_once.is_set():
            parked_once.set()
            a_in_replace.set()
            a_can_finish.wait(timeout=5.0)
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", parking_replace)

    # Writer A: enters the worker thread, parks in parking_replace.
    task_a = asyncio.create_task(store.set_nodes(nodes_a))

    for _ in range(200):
        if a_in_replace.is_set():
            break
        await asyncio.sleep(0.01)
    assert a_in_replace.is_set(), "writer A did not enter parking_replace"

    # Cancel writer A. The outer task's ``await asyncio.shield(...)``
    # re-raises CancelledError; pre-fix the surrounding async-with
    # releases the lock while the shielded inner task keeps running.
    task_a.cancel()

    # Yield so the cancel can propagate and (pre-fix) release the
    # lock on writer A's side.
    for _ in range(5):
        await asyncio.sleep(0)

    # Post-fix invariant: the lock is STILL held even though the
    # outer task A has unwound, because the shield wraps the entire
    # locked region. A concurrent caller's ``set_nodes`` will park
    # on the lock until the shielded inner write completes.
    #
    # Pre-fix: ``self._lock.locked()`` is False here — the
    # ``async with`` ``__aexit__`` released the lock while the
    # shielded write is still parked in ``parking_replace``.
    assert store._lock.locked(), (
        "after cancel of writer A, the store lock must still be held "
        "until the shielded inner write completes; pre-fix the lock "
        "was released by the surrounding ``async with``'s ``__aexit__`` "
        "while the shielded ``_write_and_publish`` continued in the "
        "background, opening a race for a concurrent second writer."
    )

    # Release writer A so it can complete and unwind.
    a_can_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task_a

    # Confirm the lock is eventually released cleanly so a follow-on
    # writer can proceed.
    for _ in range(200):
        if not store._lock.locked():
            break
        await asyncio.sleep(0.01)
    assert not store._lock.locked(), (
        "the lock must be released once the shielded inner write completes"
    )

    # And a follow-on writer completes normally.
    await store.set_nodes(nodes_b)
    on_disk = YamlNodeStore(tmp_path / "nodes.yaml")
    disk_nodes = tuple(await on_disk.get_nodes())
    in_memory = tuple(await store.get_nodes())
    assert disk_nodes == tuple(nodes_b)
    assert in_memory == disk_nodes
