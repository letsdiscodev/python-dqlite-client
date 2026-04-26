"""Pin: ``find_leader``'s shielded task observes its own exception.

When every external ``await asyncio.shield(task)`` caller is cancelled
out (e.g., by an outer ``asyncio.timeout``) before the inner task
resolves, the inner task continues per shield semantics. If it then
raises (e.g., ``ClusterError`` because every node returned no-leader),
no caller observes the exception — asyncio's GC then logs "Task
exception was never retrieved" via the loop's exception handler.

The fix calls ``task.exception()`` from the done-callback to mark the
exception observed.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_inner_exception_observed_via_done_callback_probe() -> None:
    """Direct pin via a probe done-callback registered AFTER the
    cluster's. Done-callbacks run in registration order, so by the
    time the probe runs the cluster has already (with the fix) called
    ``task.exception()``, which sets ``_log_traceback = False``. The
    probe inspects that private attribute — without the fix it would
    still be True, signalling that ``Task.__del__`` would later route
    the exception through the loop's exception handler.
    """
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    inner_started = asyncio.Event()
    inner_release = asyncio.Event()

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
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

    # Register the probe BEFORE we cancel anyone — done-callbacks
    # fire in registration order, so the cluster's _clear_slot
    # (registered first inside find_leader) runs before our probe.
    log_traceback_at_probe: list[bool] = []

    def probe(t: asyncio.Task[str]) -> None:
        # ``_log_traceback`` is a CPython private attribute; access
        # via the name-mangled form. False = exception has been
        # observed (Task.__del__ won't log). True = unobserved.
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
    # Drain done-callbacks: ``set_exception`` schedules them via
    # ``call_soon``, so they run on the *next* loop turn after
    # ``done()`` becomes True. Without this drain the probe list is
    # empty even though the task is done.
    for _ in range(5):
        await asyncio.sleep(0)
    assert log_traceback_at_probe == [False], (
        "cluster._clear_slot did not call task.exception() — "
        "Task.__del__ would log 'Task exception was never retrieved'"
    )
    # Drain so the test cleanup doesn't see an un-observed exception.
    inner_task.exception()
    gc.collect()


@pytest.mark.asyncio
async def test_success_path_unaffected() -> None:
    """Pin: the done-callback observation is harmless on the success
    path — a successful task just returns the result; the
    ``task.exception()`` call returns None and is a no-op."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
        return "elected:9001"

    cluster._find_leader_impl = _fake_impl

    leader = await cluster.find_leader()
    assert leader == "elected:9001"


@pytest.mark.asyncio
async def test_inner_exception_propagates_to_live_caller() -> None:
    """Pin: the done-callback observation must NOT swallow the
    exception from a still-live ``await asyncio.shield(task)`` —
    ``task.exception()`` is non-destructive (it returns the exception,
    does not consume it)."""
    store = MemoryNodeStore(["localhost:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    async def _fake_impl(*, trust_server_heartbeat: bool) -> str:
        raise ClusterError("simulated: no leader on any node")

    cluster._find_leader_impl = _fake_impl

    with pytest.raises(ClusterError, match="no leader"):
        await cluster.find_leader()
