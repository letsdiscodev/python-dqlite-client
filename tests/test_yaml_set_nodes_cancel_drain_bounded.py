"""Pin: ``YamlNodeStore.set_nodes``'s shield-cancel-drain loop bounds
the number of absorbed cancels at ``_MAX_CANCEL_DRAIN_ITERS``.

Without the bound, a persistent cancel-storm + a wedged worker
thread (kernel I/O hang, NFS mount lost, encrypted volume sealed)
would spin here forever holding ``self._lock`` — wedging every
other ``set_nodes`` caller on the same store process-wide. The
cap surfaces a clear ``RuntimeError`` and releases the lock so
the wedge is visible rather than silent.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from dqliteclient.node_store import (
    _MAX_CANCEL_DRAIN_ITERS,
    NodeInfo,
    YamlNodeStore,
)
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_cancel_drain_loop_caps_at_max_iters_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive a persistent cancel-storm against a parked worker. After
    ``_MAX_CANCEL_DRAIN_ITERS`` cancels are absorbed, the loop must
    raise ``RuntimeError`` with a stuck-worker diagnostic AND release
    the lock so a subsequent ``set_nodes`` is not wedged.
    """
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]

    started = threading.Event()
    can_finish = threading.Event()
    real_replace = os.replace

    def parked_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        # Park until the test explicitly releases — this stands in for
        # a wedged worker thread (NAS pause, NFS mount lost, etc.).
        can_finish.wait(timeout=30.0)
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", parked_replace)

    task = asyncio.create_task(store.set_nodes(new_nodes))

    # Wait for the worker thread to enter the parked replace.
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker did not enter parked_replace"
    assert store._lock.locked(), "lock must be held while inner runs"

    # Cancel-storm: well past the cap.
    for _ in range(_MAX_CANCEL_DRAIN_ITERS + 5):
        task.cancel()
        # Give the loop a chance to absorb each cancel.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    # The cap fires: RuntimeError surfaces with the stuck-worker
    # diagnostic. The pre-fix behaviour would have spun absorbing
    # cancels forever.
    with pytest.raises(RuntimeError, match="cancel-drain budget exceeded"):
        await task

    # Lock released so a subsequent set_nodes is not wedged.
    assert not store._lock.locked(), (
        "lock must be released after the cap fires — otherwise every "
        "subsequent set_nodes is wedged process-wide"
    )

    # Release the parked worker so the orphaned to_thread future
    # finishes cleanly and the test does not hang the runner shutdown.
    can_finish.set()


@pytest.mark.asyncio
async def test_modest_cancel_burst_below_cap_still_completes_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handful of cancels (well below the cap) must NOT trip the
    RuntimeError arm — the existing drain-loop semantics (wait for
    inner to finish, then re-raise the cancel) are preserved."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]

    started = threading.Event()
    can_finish = threading.Event()
    real_replace = os.replace

    def slow_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        started.set()
        can_finish.wait(timeout=5.0)
        real_replace(src, dst)

    monkeypatch.setattr("dqliteclient.node_store.os.replace", slow_replace)

    task = asyncio.create_task(store.set_nodes(new_nodes))

    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    # Modest burst, well below the cap.
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)

    # Release the worker — the drain loop completes normally and
    # re-raises CancelledError (no RuntimeError, no cap trip).
    can_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not store._lock.locked()
