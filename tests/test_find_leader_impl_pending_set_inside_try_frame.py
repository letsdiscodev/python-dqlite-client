"""Pin: ``ClusterClient._find_leader_impl`` builds its ``pending``
task set INSIDE the ``try:`` frame so a ``BaseException`` landing
mid-construction does not leak orphaned probe tasks.

Pre-fix the comprehension at ``cluster.py:909-911`` ran BEFORE the
``try:`` at line 916. A ``KeyboardInterrupt`` / synthetic cancel
landing in the bytecode window between the comprehension building
``pending`` and ``try:`` entering left the already-created tasks
orphaned — no done-callback observer (unlike ``find_leader``'s
``_observe_drain_exception`` discipline) — and Python emitted
"Task exception was never retrieved" / "Task was destroyed but it
is pending" warnings at GC.

The fix moves the task creation inside the try frame using a
``for``-loop so a ``KeyboardInterrupt`` raised by the n-th
``create_task`` keeps every preceding task in ``pending`` for the
``finally`` to cancel + gather.

This test mirrors the ``done/ISSUE-243_pool-acquire-orphans-tasks-on-pre-try-cancel.md``
hardening on the pool side.
"""

from __future__ import annotations

import asyncio
import gc
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_find_leader_impl_keyboardinterrupt_during_task_creation_no_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject ``KeyboardInterrupt`` into the n-th ``create_task`` call
    inside ``_find_leader_impl`` so the first task is created, then the
    comprehension/loop aborts. Pin: no orphan-task warnings via
    ``loop.set_exception_handler``."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    # Make every probe coroutine raise a non-suppressed programming-bug
    # exception so an orphaned task (one created but never gathered)
    # surfaces a "Task exception was never retrieved" context. Without
    # this the probe would block on a dial against a fake address; an
    # orphaned task simply parked forever wouldn't trigger the
    # exception handler.
    cluster._query_leader = AsyncMock(side_effect=TypeError("synthetic-probe-bug"))

    real_create_task = asyncio.create_task

    # The cluster module does ``import asyncio`` and references
    # ``asyncio.create_task`` — monkeypatch the attribute on the
    # cluster module's bound ``asyncio`` reference. That's the
    # symbol resolved at call time.
    import dqliteclient.cluster as cluster_mod

    call_count = {"n": 0}
    created: list[asyncio.Task[object]] = []

    def patched_create_task(coro):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            t = real_create_task(coro)
            created.append(t)
            return t
        # 2nd call: simulate a synthetic BaseException landing in the
        # bytecode window of the comprehension. Coro must be closed
        # to avoid an "RuntimeWarning: coroutine ... was never awaited".
        coro.close()
        raise KeyboardInterrupt("synthetic")

    monkeypatch.setattr(cluster_mod.asyncio, "create_task", patched_create_task)

    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        await cluster._find_leader_impl(trust_server_heartbeat=False)

    # The first task was created BEFORE the synthetic KeyboardInterrupt.
    # Post-fix discipline: the task is built inside the try frame, so
    # the ``finally:`` cancels-and-gathers it before the
    # KeyboardInterrupt propagates. Pre-fix: comprehension ran outside
    # the try, so no finally executed, and the task is still alive
    # (running or in a completed-but-unobserved state).
    assert len(created) == 1, "test wired wrong: expected exactly one task created"
    leaked = created[0]
    if not leaked.done():
        # Defensive cleanup: silence a real leak so the suite stays
        # clean, then assert what we actually observed.
        leaked.cancel()
        with pytest.raises((asyncio.CancelledError, TypeError)):
            await leaked
        pytest.fail(
            "_find_leader_impl orphaned probe task: not done after caller "
            "saw KeyboardInterrupt — comprehension is outside the try frame"
        )
    # Task is done. Pin: its exception was observed (i.e. the finally's
    # ``asyncio.gather(*pending, return_exceptions=True)`` ran and
    # consumed the result). The defining post-fix invariant.
    captured: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        # Drop our strong ref + force GC so any unobserved task
        # exception fires now, while the handler is bound.
        del leaked
        created.clear()
        gc.collect()
        for _ in range(3):
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_handler)

    orphan_msgs = [
        ctx
        for ctx in captured
        if any(
            marker in str(ctx.get("message", ""))
            for marker in ("Task exception was never retrieved", "Task was destroyed")
        )
    ]
    assert not orphan_msgs, (
        f"orphaned probe tasks observed via loop exception handler: {orphan_msgs}"
    )
