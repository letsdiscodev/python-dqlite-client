"""``_find_leader_impl`` builds its ``pending`` task set inside the
``try:`` frame so a ``BaseException`` landing mid-construction leaves
every created task in ``pending`` for the ``finally`` to cancel+gather,
rather than orphaning probe tasks."""

from __future__ import annotations

import asyncio
import gc
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_find_leader_impl_keyboardinterrupt_during_task_creation_no_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject ``KeyboardInterrupt`` into the 2nd ``create_task`` so the
    first task is created then construction aborts; assert no orphan-task
    warnings via ``loop.set_exception_handler``."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    # Probes raise a non-suppressed bug exception so an orphaned (never
    # gathered) task surfaces a "Task exception was never retrieved"
    # context; a task merely parked on a dial would not.
    cluster._query_leader = AsyncMock(side_effect=TypeError("synthetic-probe-bug"))

    real_create_task = asyncio.create_task

    import dqliteclient.cluster as cluster_mod

    call_count = {"n": 0}
    created: list[asyncio.Task[object]] = []

    def patched_create_task(coro: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            t = real_create_task(coro)
            created.append(t)
            return t
        # Close the coro to avoid "coroutine ... was never awaited".
        coro.close()
        raise KeyboardInterrupt("synthetic")

    monkeypatch.setattr("asyncio.create_task", patched_create_task)
    _ = cluster_mod  # keep import for module-load side effects

    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        await cluster._find_leader_impl(trust_server_heartbeat=False)

    assert len(created) == 1, "test wired wrong: expected exactly one task created"
    leaked = created[0]
    if not leaked.done():
        # Silence a real leak, then assert what we observed.
        leaked.cancel()
        with pytest.raises((asyncio.CancelledError, TypeError)):
            await leaked
        pytest.fail(
            "_find_leader_impl orphaned probe task: not done after caller "
            "saw KeyboardInterrupt — comprehension is outside the try frame"
        )
    # Task is done; its exception must have been observed by the finally's
    # gather(..., return_exceptions=True).
    captured: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        # Drop the strong ref and force GC so any unobserved task
        # exception fires while the handler is bound.
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
