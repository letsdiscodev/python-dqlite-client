"""``find_leader`` collapses concurrent callers onto a single in-flight
discovery task, turning N independent sweeps into one."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_concurrent_find_leader_callers_share_one_task() -> None:
    """20 concurrent callers must collapse onto one ``_find_leader_impl``
    execution."""
    store = MemoryNodeStore(["localhost:9001", "localhost:9002", "localhost:9003"])
    cluster = ClusterClient(store, timeout=0.5)

    impl_calls = 0
    leader_event = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        nonlocal impl_calls
        impl_calls += 1
        await leader_event.wait()  # park so all 20 callers stack up
        return "elected:9001"

    cluster._find_leader_impl = _fake_impl

    callers = [asyncio.create_task(cluster.find_leader()) for _ in range(20)]
    for _ in range(10):
        await asyncio.sleep(0)

    leader_event.set()
    results = await asyncio.gather(*callers)

    assert all(r == "elected:9001" for r in results)
    assert impl_calls == 1, f"expected one shared sweep across 20 callers; ran {impl_calls}"


@pytest.mark.asyncio
async def test_consecutive_find_leader_calls_get_fresh_probes() -> None:
    """The slot clears on completion, so consecutive callers run
    independent probes; failures are not cached."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    impl_calls = 0

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        nonlocal impl_calls
        impl_calls += 1
        return f"leader-{impl_calls}:9001"

    cluster._find_leader_impl = _fake_impl

    a = await cluster.find_leader()
    b = await cluster.find_leader()
    c = await cluster.find_leader()
    assert a == "leader-1:9001"
    assert b == "leader-2:9001"
    assert c == "leader-3:9001"
    assert impl_calls == 3
