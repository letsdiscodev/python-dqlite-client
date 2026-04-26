"""Pin: ``_clear_slot`` handles a cancelled inner task without
calling ``task.exception()`` and still clears the slot.

The done-callback's ``not t.cancelled()`` guard exists because
calling ``Task.exception()`` on a cancelled task raises
``CancelledError`` (asyncio docs). Without the guard, the suppressed
``BaseException`` arm catches the CancelledError but the slot
clearing has already happened — the visible failure mode would be
spurious "Task was cancelled" logs through the loop's exception
handler if we ever stopped suppressing BaseException.

The slot must still be cleared so the next ``find_leader`` triggers
a fresh probe instead of awaiting the dead cancelled task.
"""

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

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
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

    # Pin 1: slot was cleared even on the cancelled arm.
    assert cluster._find_leader_tasks == {}
    # Pin 2: the inner task remains cancelled (we did not accidentally
    # observe-then-resume it via exception()).
    assert inner_task.cancelled()


@pytest.mark.asyncio
async def test_cancelled_inner_task_next_caller_starts_fresh_probe() -> None:
    """Follow-up: after a cancelled inner clears the slot, the next
    ``find_leader`` must spawn a NEW task instead of awaiting the dead
    one. Without the slot-clear, the next caller would await a
    cancelled task and immediately re-raise CancelledError."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    call_count = 0
    inner_started = asyncio.Event()
    inner_release = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
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

    # Next caller: should spawn a fresh probe and resolve cleanly.
    leader = await cluster.find_leader()
    assert leader == "second:9001"
    assert call_count == 2
