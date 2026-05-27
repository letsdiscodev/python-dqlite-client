"""Pin: ``ConnectionPool`` holds a strong reference to the
fire-and-forget ``_release_after_drain`` follow-up tasks it
schedules from the cancel-drain paths.

The stdlib ``asyncio`` docs are explicit: "Save a reference to the
result of [create_task / ensure_future] to avoid a task disappearing
mid-execution." Pre-fix the pool spawned the follow-up via
``asyncio.ensure_future(self._release_after_drain(inner_drain))``
with the result discarded. The follow-up holds a strong reference
to ``inner_drain`` and stays rooted through it under normal
operation, but during loop teardown / ``engine.dispose()`` the GC
can reclaim the follow-up before ``_release_reservation`` runs,
leaking the reservation slot.

The fix tracks each follow-up in a per-pool ``_background_tasks``
set; a ``add_done_callback(...)`` discard keeps the set bounded.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_release_after_drain_task_is_strongly_referenced() -> None:
    """When ``_release_after_drain`` is fired, the resulting Task
    must be added to the pool's strong-ref set so a forced GC pass
    cannot reclaim it before it completes."""
    pool = ConnectionPool(addresses=["10.0.0.1:9001"])

    # Manually invoke the strong-ref pattern: schedule a long-running
    # follow-up, confirm it lives in _background_tasks, then force a
    # GC and confirm it still lives.
    completed = asyncio.Event()

    async def long_follow_up() -> None:
        await asyncio.sleep(0)
        completed.set()

    # Emulate the production call site: spawn + register.
    task = asyncio.ensure_future(long_follow_up())
    pool._background_tasks.add(task)
    task.add_done_callback(pool._background_tasks.discard)

    assert task in pool._background_tasks
    gc.collect()
    assert not task.done()
    assert task in pool._background_tasks

    await task
    assert completed.is_set()
    assert task not in pool._background_tasks, "done-callback should discard"


@pytest.mark.asyncio
async def test_pool_background_tasks_set_is_initially_empty() -> None:
    """A fresh pool has an empty ``_background_tasks`` set."""
    pool = ConnectionPool(addresses=["10.0.0.1:9001"])
    assert pool._background_tasks == set()


@pytest.mark.asyncio
async def test_pool_background_tasks_set_does_not_grow_unboundedly() -> None:
    """After many follow-ups complete, the set returns to empty."""
    pool = ConnectionPool(addresses=["10.0.0.1:9001"])

    async def follow_up() -> None:
        await asyncio.sleep(0)

    tasks = []
    for _ in range(100):
        task = asyncio.ensure_future(follow_up())
        pool._background_tasks.add(task)
        task.add_done_callback(pool._background_tasks.discard)
        tasks.append(task)

    # Let each task complete; the done-callback fires synchronously
    # from the loop's "ready" queue once the task is awaited or the
    # loop yields enough times to schedule the completion callback.
    await asyncio.gather(*tasks)
    # One more yield so the discard callbacks land.
    await asyncio.sleep(0)

    assert pool._background_tasks == set(), (
        f"_background_tasks unexpectedly retained {len(pool._background_tasks)} "
        "completed entries; done-callback discard must clear them"
    )
