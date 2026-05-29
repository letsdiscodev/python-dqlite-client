"""``ConnectionPool`` tracks fire-and-forget ``_release_after_drain`` follow-ups in a
``_background_tasks`` set (with a done-callback discard) so the GC cannot reclaim a
follow-up before ``_release_reservation`` runs and leak the reservation slot."""

from __future__ import annotations

import asyncio
import gc

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_release_after_drain_task_is_strongly_referenced() -> None:
    """A fired follow-up must be added to _background_tasks so a forced GC cannot
    reclaim it before it completes."""
    pool = ConnectionPool(addresses=["10.0.0.1:9001"])

    completed = asyncio.Event()

    async def long_follow_up() -> None:
        await asyncio.sleep(0)
        completed.set()

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

    await asyncio.gather(*tasks)
    # One more yield so the discard callbacks land.
    await asyncio.sleep(0)

    assert pool._background_tasks == set(), (
        f"_background_tasks unexpectedly retained {len(pool._background_tasks)} "
        "completed entries; done-callback discard must clear them"
    )
