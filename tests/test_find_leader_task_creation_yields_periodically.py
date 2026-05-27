"""Pin: ``_find_leader_impl``'s probe-task creation loop yields to
the loop scheduler every ``_PROBE_TASK_CREATE_YIELD_EVERY``
allocations so a hostile or buggy NodeStore approaching the wire
cap (10_000 entries) cannot monopolise the loop during the
allocation burst.

The bound is on loop yielding, not on in-flight task count: the
parallel verify-redirect optimisation in ``_probe_one`` requires
all tasks to be created up-front so the post-semaphore verify
phase can overlap across nodes (see the semaphore-scope comment
in ``_probe_one``). Gating task creation behind the semaphore would
serialise verifies and defeat that optimisation; yielding every K
allocations keeps the loop responsive without changing the
verify-overlap semantics.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import _PROBE_TASK_CREATE_YIELD_EVERY, ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_probe_task_create_yield_constant_in_sensible_range() -> None:
    """The constant must be in a sensible band: not so small that
    every iteration yields (overhead), not so large that the burst
    is unbounded."""
    assert 16 <= _PROBE_TASK_CREATE_YIELD_EVERY <= 4096


@pytest.mark.asyncio
async def test_find_leader_task_creation_yields_to_scheduler_during_burst() -> None:
    """With N=2000 nodes, the task-creation loop must call
    ``asyncio.sleep(0)`` (cede a scheduler tick) at least
    ``N / _PROBE_TASK_CREATE_YIELD_EVERY`` times during the
    allocation burst. Without that cooperative yield, the create
    loop monopolises the loop until the wait/asyncio scheduler
    boundary."""
    addresses = [f"10.0.0.{i // 256}.{i % 256}:9001" for i in range(2000)]
    cluster = ClusterClient(
        MemoryNodeStore(addresses),
        timeout=5.0,
        concurrent_leader_conns=10,
    )

    # Park each probe forever so the create loop is observed in
    # isolation from probe execution.
    park_forever = asyncio.Event()

    async def _query_leader_park(addr: str, **_kw: object) -> str | None:
        await park_forever.wait()
        return None

    # Count the number of ``asyncio.sleep(0)`` calls that originate
    # from inside ``dqliteclient.cluster``. The create loop is the
    # only such call site that fires per N/K during the find-leader
    # sweep before any probe runs through ``_query_leader`` (probes
    # are parked).
    sleep_zero_calls = 0
    original_sleep = asyncio.sleep

    async def counting_sleep(delay: float, *args: object, **kwargs: object) -> None:
        nonlocal sleep_zero_calls
        if delay == 0:
            sleep_zero_calls += 1
        await original_sleep(delay, *args, **kwargs)

    with (
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_query_leader_park)),
        patch("dqliteclient.cluster.asyncio.sleep", new=counting_sleep),
    ):
        sweep_task = asyncio.create_task(cluster.find_leader())
        # Give the create loop time to run to completion. Each yield
        # inside the create loop returns control here; we keep
        # waiting until the sweep is parked on asyncio.wait. The
        # tick budget is generous: ~2 ticks per expected yield plus
        # slack for scheduler boundaries.
        max_ticks = max(200, (2000 // _PROBE_TASK_CREATE_YIELD_EVERY) * 4)
        for _ in range(max_ticks):
            await original_sleep(0)
        sweep_task.cancel()
        park_forever.set()
        with pytest.raises(asyncio.CancelledError):
            await sweep_task

    expected_min = (2000 // _PROBE_TASK_CREATE_YIELD_EVERY) - 1
    assert sleep_zero_calls >= expected_min, (
        f"create loop made only {sleep_zero_calls} sleep(0) calls; "
        f"expected at least {expected_min} for N=2000 nodes / "
        f"yield-every={_PROBE_TASK_CREATE_YIELD_EVERY}"
    )
