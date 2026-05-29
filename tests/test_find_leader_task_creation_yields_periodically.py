"""Pin: the probe-task creation loop yields every ``_PROBE_TASK_CREATE_YIELD_EVERY``
allocations so a NodeStore near the 10_000 wire cap can't monopolise the loop.

Tasks are created up-front (not gated behind the semaphore) so the post-semaphore
verify phase can overlap across nodes; yielding every K keeps that overlap intact."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import _PROBE_TASK_CREATE_YIELD_EVERY, ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_probe_task_create_yield_constant_in_sensible_range() -> None:
    """The constant stays in a sensible band: not per-iteration, not unbounded."""
    assert 16 <= _PROBE_TASK_CREATE_YIELD_EVERY <= 4096


@pytest.mark.asyncio
async def test_find_leader_task_creation_yields_to_scheduler_during_burst() -> None:
    """With N=2000 nodes the create loop must call ``asyncio.sleep(0)`` at least
    ``N / _PROBE_TASK_CREATE_YIELD_EVERY`` times during the allocation burst."""
    addresses = [f"10.0.0.{i // 256}.{i % 256}:9001" for i in range(2000)]
    cluster = ClusterClient(
        MemoryNodeStore(addresses),
        timeout=5.0,
        concurrent_leader_conns=10,
    )

    # Park each probe forever so the create loop is observed in isolation.
    park_forever = asyncio.Event()

    async def _query_leader_park(addr: str, **_kw: object) -> str | None:
        await park_forever.wait()
        return None

    # The create loop is the only ``sleep(0)`` site that fires before any parked
    # probe runs, so counting them measures create-loop yields.
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
        # Generous tick budget (~2 ticks per expected yield plus slack) to let
        # the create loop run to completion before the sweep parks on asyncio.wait.
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
