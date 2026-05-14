"""Pin: ``ConnectionPool.initialize`` builds its ``create_tasks``
list INSIDE the ``try:`` frame so a ``BaseException`` (synthetic
KeyboardInterrupt, outer cancel) landing mid-construction does not
leak orphaned ``_create_connection`` tasks.

The anchor is the in-try comment block beginning
``"Build ``create_tasks`` INSIDE the try frame"`` inside
``ConnectionPool.initialize`` in ``pool.py``; the construction
itself is the ``create_tasks: list[...] = []`` allocation followed
by the ``for _ in range(self._min_size): create_tasks.append(...)``
loop, both nested under the same ``try:``.

Pre-fix this construction (then written as a list comprehension,
now an explicit ``for`` loop) ran BEFORE the ``try:`` frame. A
``BaseException`` raised by the n-th ``asyncio.create_task``
orphaned every already-created task — the ``finally`` never ran,
the ``gather_returned``-flag-driven recovery loop never walked the
partial list, and the orphan tasks' exceptions were unobserved.

Mirrors the cluster-side hardening at ``_find_leader_impl`` and the
pool-acquire hardening in
``done/ISSUE-243_pool-acquire-orphans-tasks-on-pre-try-cancel.md``.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_initialize_keyboardinterrupt_during_task_creation_no_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject ``KeyboardInterrupt`` into the n-th ``create_task`` call
    inside ``ConnectionPool.initialize``. The first task is created;
    the ``for`` loop aborts before ``gather`` can run. Pin: the
    task is consumed by the ``finally`` (cancelled + gathered), no
    orphan-task warnings."""
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    # Force the orphaned task to surface a programming-bug exception
    # so the unobserved-exception case is detectable. A successful
    # _create_connection would silently return a connection that
    # never gets closed — also a leak, but harder to pin via the
    # loop exception handler.
    pool._create_connection = AsyncMock(
        side_effect=TypeError("synthetic-create-bug"),
    )

    real_create_task = asyncio.create_task

    import dqliteclient.pool as pool_mod

    call_count = {"n": 0}
    created: list[asyncio.Task[object]] = []

    def patched_create_task(coro: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            t = real_create_task(coro)
            created.append(t)
            return t
        coro.close()
        raise KeyboardInterrupt("synthetic")

    monkeypatch.setattr("asyncio.create_task", patched_create_task)
    _ = pool_mod  # keep import for module-load side effects

    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        await pool.initialize()

    assert len(created) == 1, "test wired wrong: expected exactly one task created"
    leaked = created[0]
    if not leaked.done():
        # Defensive cleanup so the suite stays clean, then fail.
        leaked.cancel()
        with pytest.raises((asyncio.CancelledError, TypeError)):
            await leaked
        pytest.fail(
            "ConnectionPool.initialize orphaned a _create_connection task: "
            "task construction is outside the try frame"
        )

    # Pin no orphan-task warnings via the loop exception handler.
    captured: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
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
    assert not orphan_msgs, f"orphaned create_connection tasks observed: {orphan_msgs}"
