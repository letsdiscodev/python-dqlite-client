"""``ClusterClient.find_leader`` collapses concurrent callers onto a
single in-flight discovery task. Under a leader flip with N waiting
acquirers, this turns N independent per-node sweeps into one.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_concurrent_find_leader_callers_share_one_task() -> None:
    """20 concurrent callers must collapse onto a single
    ``_find_leader_impl`` execution. Pin the count via a counter on
    the impl wrapper."""
    store = MemoryNodeStore(["localhost:9001", "localhost:9002", "localhost:9003"])
    cluster = ClusterClient(store, timeout=0.5)

    impl_calls = 0
    leader_event = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
        nonlocal impl_calls
        impl_calls += 1
        # Park until released so all 20 callers stack up on the
        # shared task.
        await leader_event.wait()
        return "elected:9001"

    cluster._find_leader_impl = _fake_impl  # type: ignore[method-assign]

    callers = [asyncio.create_task(cluster.find_leader()) for _ in range(20)]
    # Yield enough times for every caller to enter find_leader.
    for _ in range(10):
        await asyncio.sleep(0)

    leader_event.set()
    results = await asyncio.gather(*callers)

    assert all(r == "elected:9001" for r in results)
    assert impl_calls == 1, f"expected one shared sweep across 20 callers; ran {impl_calls}"


@pytest.mark.asyncio
async def test_consecutive_find_leader_calls_get_fresh_probes() -> None:
    """The single-flight slot clears when the current task completes,
    so consecutive callers run independent probes — failures are NOT
    cached."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    impl_calls = 0

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
        nonlocal impl_calls
        impl_calls += 1
        return f"leader-{impl_calls}:9001"

    cluster._find_leader_impl = _fake_impl  # type: ignore[method-assign]

    a = await cluster.find_leader()
    b = await cluster.find_leader()
    c = await cluster.find_leader()
    assert a == "leader-1:9001"
    assert b == "leader-2:9001"
    assert c == "leader-3:9001"
    assert impl_calls == 3
