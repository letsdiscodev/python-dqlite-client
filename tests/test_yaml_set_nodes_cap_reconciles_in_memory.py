"""``set_nodes``'s cancel-cap fire path re-loads ``self._nodes`` from disk so the
in-memory snapshot matches on-disk truth (the worker may have finalised the rename
after we gave up)."""

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
    """After the cap-fire RuntimeError, ``get_nodes()`` reflects disk, not a stale tuple."""
    store = YamlNodeStore(tmp_path / "nodes.yaml")
    initial = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
    ]
    await store.set_nodes(initial)
    assert tuple(await store.get_nodes()) == tuple(initial)

    # Worker writes to disk then parks: disk holds the NEW payload while
    # self._nodes still holds the OLD value (the cancel landed first). This
    # is the divergence the cap-fire re-load must repair.
    parked = asyncio.Event()

    async def _write_then_park(
        nodes: tuple[NodeInfo, ...], payload: list[dict[str, object]]
    ) -> None:
        # Mirror the real signature: serialise inside the worker thread.
        import yaml

        text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        await asyncio.to_thread(lambda: (tmp_path / "nodes.yaml").write_text(text))
        # Disk now has the new payload; self._nodes is still old. Park so the
        # cancel storm catches us with disk ahead.
        await parked.wait()

    original_write = store._write_and_publish
    store._write_and_publish = _write_then_park  # type: ignore[assignment]

    new_nodes = [
        NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="127.0.0.1:9002", role=NodeRole.VOTER),
    ]

    task = asyncio.create_task(store.set_nodes(new_nodes))
    await asyncio.sleep(0.005)  # let task acquire lock + enter shield

    for _ in range(_MAX_CANCEL_DRAIN_ITERS + 5):
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="cancel-drain budget exceeded"):
        await task

    # Restore and release the parked write so the orphan task doesn't hang teardown.
    store._write_and_publish = original_write
    parked.set()

    # The cap-fire arm schedules the reconcile as a fire-and-forget Task (the
    # cancelling state prevents an in-line await); settle the loop so it runs.
    for _ in range(50):
        await asyncio.sleep(0.005)
        current = tuple(await store.get_nodes())
        if current == tuple(initial):
            break

    # The parked worker wrote new_nodes before parking, so get_nodes() must
    # reconcile to new_nodes.
    expected = tuple(new_nodes)
    for _ in range(50):
        await asyncio.sleep(0.01)
        current = tuple(await store.get_nodes())
        if current == expected:
            break
    assert current == expected, (
        f"set_nodes cap-fire must re-sync the in-memory snapshot with "
        f"disk truth; got in-memory={current!r}, expected={expected!r}"
    )
