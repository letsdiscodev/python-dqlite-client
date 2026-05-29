"""``find_leader``'s shielded task observes its own exception via a
done-callback so a cancelled-out caller doesn't leave asyncio's GC logging
"Task exception was never retrieved"."""

from __future__ import annotations

import asyncio
import gc

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_inner_exception_observed_via_done_callback_probe() -> None:
    """Probe done-callback registered after the cluster's reads
    ``_log_traceback``: False means the cluster called ``task.exception()``
    (fixed); True means ``Task.__del__`` would still log it."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    inner_started = asyncio.Event()
    inner_release = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        inner_started.set()
        await inner_release.wait()
        raise ClusterError("simulated: no leader on any node")

    cluster._find_leader_impl = _fake_impl

    async def cancel_after_inner_starts() -> None:
        async with asyncio.timeout(10):
            await cluster.find_leader()

    caller = asyncio.create_task(cancel_after_inner_starts())
    await inner_started.wait()
    inner_task = next(iter(cluster._find_leader_tasks.values()))

    # Register before cancelling: done-callbacks fire in registration
    # order, so the cluster's _clear_slot runs before this probe.
    log_traceback_at_probe: list[bool] = []

    def probe(t: asyncio.Task[str]) -> None:
        log_traceback_at_probe.append(getattr(t, "_log_traceback", True))

    inner_task.add_done_callback(probe)

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    inner_release.set()
    for _ in range(20):
        if inner_task.done():
            break
        await asyncio.sleep(0)
    assert inner_task.done()
    # Drain done-callbacks: set_exception schedules them via call_soon,
    # so they run on the next loop turn after done() becomes True.
    for _ in range(5):
        await asyncio.sleep(0)
    assert log_traceback_at_probe == [False], (
        "cluster._clear_slot did not call task.exception() — "
        "Task.__del__ would log 'Task exception was never retrieved'"
    )
    # Drain so test cleanup doesn't see an un-observed exception.
    inner_task.exception()
    gc.collect()


@pytest.mark.asyncio
async def test_success_path_unaffected() -> None:
    """The done-callback observation is a no-op on the success path."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        return "elected:9001"

    cluster._find_leader_impl = _fake_impl

    leader = await cluster.find_leader()
    assert leader == "elected:9001"


@pytest.mark.asyncio
async def test_inner_exception_propagates_to_live_caller() -> None:
    """The done-callback observation must not swallow the exception from a
    still-live caller — ``task.exception()`` is non-destructive."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    async def _fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        raise ClusterError("simulated: no leader on any node")

    cluster._find_leader_impl = _fake_impl

    with pytest.raises(ClusterError, match="no leader"):
        await cluster.find_leader()
