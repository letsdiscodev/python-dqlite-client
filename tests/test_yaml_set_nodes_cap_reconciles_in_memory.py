"""Pin: ``YamlNodeStore.set_nodes``'s cancel-cap fire path
re-loads ``self._nodes`` from disk so the in-memory snapshot
matches the on-disk truth (which may be the new payload — if the
worker thread finalised ``os.replace`` after we gave up — the old
payload if the worker never reached the rename, or torn if the
worker was killed mid-rename).

Pre-fix, the cap-fire raised ``RuntimeError`` without re-loading;
``self._nodes`` lagged behind disk indefinitely (until the next
successful set_nodes or process restart). Go's
``client/store_yaml.go::SetServers`` uses an uninterruptible
mutex so the divergence is unreachable there.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dqliteclient.node_store import (
    _MAX_CANCEL_DRAIN_ITERS,
    NodeInfo,
    YamlNodeStore,
)
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_cap_fire_reloads_in_memory_snapshot_from_disk(
    tmp_path: Path,
) -> None:
    """Drive set_nodes into the cap-fire path. After RuntimeError
    surfaces, ``get_nodes()`` must reflect whatever IS on disk —
    not a stale in-memory tuple."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    initial = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]
    await store.set_nodes(initial)
    assert tuple(await store.get_nodes()) == tuple(initial)

    # Patch _write_and_publish so the worker WRITES TO DISK then
    # parks indefinitely. This produces the actual divergence the
    # cap-fire's re-load is supposed to repair: disk reflects the
    # NEW payload while self._nodes still holds the OLD value
    # because the asyncio side never observed the to_thread future
    # resolving (we got cancelled first).
    parked = asyncio.Event()

    async def _write_then_park(nodes: tuple[NodeInfo, ...], text: str) -> None:
        # Drive the on-disk write synchronously, then park.
        await asyncio.to_thread(lambda: (tmp_path / "nodes.yaml").write_text(text))
        # Now the disk has the new payload. self._nodes is still old.
        # Park so the cancel storm catches us with disk ahead.
        await parked.wait()

    original_write = store._write_and_publish
    store._write_and_publish = _write_then_park  # type: ignore[assignment]

    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="127.0.0.1:9002", role=NodeRole.VOTER),
    ]

    task = asyncio.create_task(store.set_nodes(new_nodes))
    # Let task acquire lock + enter shield.
    await asyncio.sleep(0.005)

    # Cancel-storm past the cap.
    for _ in range(_MAX_CANCEL_DRAIN_ITERS + 5):
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="cancel-drain budget exceeded"):
        await task

    # Restore for cleanup; release the parked write so the orphan
    # task doesn't hang teardown.
    store._write_and_publish = original_write
    parked.set()

    # Wait for the reconcile-from-disk Task to finish. The cap-fire
    # arm schedules it as a fire-and-forget Task (the cancelling
    # state of the original task prevents an in-line await). Settle
    # the loop a few times so the to_thread worker can complete.
    for _ in range(50):
        await asyncio.sleep(0.005)
        current = tuple(await store.get_nodes())
        if current == tuple(initial):
            break

    # Load-bearing: get_nodes() reflects what disk holds. In this
    # scenario the parked worker wrote new_nodes to disk before
    # parking, so the in-memory snapshot must reconcile to new_nodes.
    expected = tuple(new_nodes)
    # Try a few times to allow the fire-and-forget reload to run.
    for _ in range(50):
        await asyncio.sleep(0.01)
        current = tuple(await store.get_nodes())
        if current == expected:
            break
    assert current == expected, (
        f"set_nodes cap-fire must re-sync the in-memory snapshot with "
        f"disk truth; got in-memory={current!r}, expected={expected!r}"
    )
