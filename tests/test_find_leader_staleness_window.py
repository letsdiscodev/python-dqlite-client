"""Pin: ``ClusterClient.find_leader`` snapshots the node store once at
the start of a sweep. A concurrent ``store.set_nodes(...)`` does NOT
affect the in-flight sweep — the staleness window is intentional and
documented at ``cluster.py:182-191``.

The next sweep (after the in-flight one completes) does observe the
update.

These tests pin both halves of the contract so a future refactor that
e.g. re-reads the store mid-sweep, or that propagates updates into a
running sweep, has to be a deliberate behavior change.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _node(addr: str, node_id: int = 1) -> NodeInfo:
    return NodeInfo(node_id=node_id, address=addr, role=NodeRole.VOTER)


@pytest.mark.asyncio
async def test_in_flight_sweep_does_not_observe_concurrent_set_nodes() -> None:
    """A concurrent ``set_nodes`` during an in-flight sweep does not
    rewrite the snapshot the sweep is iterating."""
    store = MemoryNodeStore(["a:9001", "b:9002"])
    cluster = ClusterClient(store, timeout=0.5)

    snapshots: list[list[str]] = []
    parked = asyncio.Event()
    release = asyncio.Event()

    async def _impl(*, trust_server_heartbeat: bool) -> str:
        nodes = await store.get_nodes()
        snapshots.append([n.address for n in nodes])
        parked.set()
        await release.wait()
        return "leader:9001"

    cluster._find_leader_impl = _impl

    sweep = asyncio.create_task(cluster.find_leader())
    await parked.wait()

    # Concurrent update — must not affect the in-flight sweep.
    await store.set_nodes([_node("c:9003", 3), _node("d:9004", 4)])

    release.set()
    result = await sweep
    assert result == "leader:9001"

    # The first (in-flight) sweep saw the original list.
    assert snapshots[0] == ["a:9001", "b:9002"], (
        "in-flight sweep must use its initial node-store snapshot, not"
        f" observe a concurrent set_nodes; saw {snapshots[0]}"
    )


@pytest.mark.asyncio
async def test_next_sweep_after_set_nodes_observes_update() -> None:
    """The single-flight cache clears after a sweep completes; the
    next call observes the updated node store."""
    store = MemoryNodeStore(["a:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    snapshots: list[list[str]] = []

    async def _impl(*, trust_server_heartbeat: bool) -> str:
        nodes = await store.get_nodes()
        snapshots.append([n.address for n in nodes])
        return f"leader:from-{snapshots[-1][0]}"

    cluster._find_leader_impl = _impl

    first = await cluster.find_leader()
    await store.set_nodes([_node("x:9999", 9)])
    second = await cluster.find_leader()

    assert snapshots[0] == ["a:9001"]
    assert snapshots[1] == ["x:9999"], (
        "next sweep after set_nodes must observe the updated node list"
    )
    assert first == "leader:from-a:9001"
    assert second == "leader:from-x:9999"


@pytest.mark.asyncio
async def test_late_callers_during_sweep_see_inflight_snapshot_too() -> None:
    """Single-flight collapses concurrent callers; a caller that
    arrives mid-sweep waits on the same task and inherits its snapshot
    — even if ``set_nodes`` was called after they entered.
    """
    store = MemoryNodeStore(["a:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    impl_calls = 0
    parked = asyncio.Event()
    release = asyncio.Event()

    async def _impl(**_: Any) -> str:
        nonlocal impl_calls
        impl_calls += 1
        parked.set()
        await release.wait()
        return "leader:9001"

    cluster._find_leader_impl = _impl

    first = asyncio.create_task(cluster.find_leader())
    await parked.wait()

    await store.set_nodes([_node("x:9999", 9)])
    second = asyncio.create_task(cluster.find_leader())
    # Yield once so ``second`` enters and joins the in-flight sweep.
    await asyncio.sleep(0)

    release.set()
    a, b = await asyncio.gather(first, second)

    assert a == b == "leader:9001"
    assert impl_calls == 1, (
        "the late caller must collapse onto the in-flight sweep, not "
        "trigger a fresh sweep with the updated nodes"
    )
