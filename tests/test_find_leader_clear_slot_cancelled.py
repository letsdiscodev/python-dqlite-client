"""``_clear_slot`` handles a cancelled inner task without calling
``task.exception()`` (which would re-raise ``CancelledError``) and still
clears the slot so the next ``find_leader`` triggers a fresh probe."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_cancelled_inner_task_clears_slot_without_calling_exception() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    inner_started = asyncio.Event()
    inner_release = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        inner_started.set()
        await inner_release.wait()
        return "ignored:9001"

    cluster._find_leader_impl = _fake_impl

    caller = asyncio.create_task(cluster.find_leader())
    await inner_started.wait()
    inner_task = next(iter(cluster._find_leader_tasks.values()))

    inner_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # Drain the done-callback chain.
    for _ in range(5):
        await asyncio.sleep(0)

    assert cluster._find_leader_tasks == {}
    assert inner_task.cancelled()


@pytest.mark.asyncio
async def test_cancelled_inner_task_next_caller_starts_fresh_probe() -> None:
    """After a cancelled inner clears the slot, the next ``find_leader``
    must spawn a new task instead of awaiting the dead one."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    call_count = 0
    inner_started = asyncio.Event()
    inner_release = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            inner_started.set()
            await inner_release.wait()
            return "first:9001"
        return "second:9001"

    cluster._find_leader_impl = _fake_impl

    caller = asyncio.create_task(cluster.find_leader())
    await inner_started.wait()
    inner_task = next(iter(cluster._find_leader_tasks.values()))
    inner_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    for _ in range(5):
        await asyncio.sleep(0)

    leader = await cluster.find_leader()
    assert leader == "second:9001"
    assert call_count == 2
