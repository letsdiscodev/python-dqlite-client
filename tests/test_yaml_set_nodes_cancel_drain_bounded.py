"""``set_nodes``'s cancel-drain loop caps absorbed cancels at ``_MAX_CANCEL_DRAIN_ITERS``,
so a cancel-storm against a wedged worker raises ``RuntimeError`` instead of spinning
forever holding the lock."""

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
    """Cancel-storm a parked worker: after the cap, the loop raises ``RuntimeError``
    and releases the lock so a subsequent ``set_nodes`` is not wedged."""
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
        # Park until released — stands in for a wedged worker thread.
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
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="cancel-drain budget exceeded"):
        await task

    # Lock released so a subsequent set_nodes is not wedged.
    assert not store._lock.locked(), (
        "lock must be released after the cap fires — otherwise every "
        "subsequent set_nodes is wedged process-wide"
    )

    # Release the parked worker so the orphaned future finishes and teardown is clean.
    can_finish.set()


@pytest.mark.asyncio
async def test_modest_cancel_burst_below_cap_still_completes_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancels below the cap must not trip the RuntimeError arm — drain then re-raise."""
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

    can_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not store._lock.locked()
