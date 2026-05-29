"""``cluster_info``'s redirect-policy filter yields periodically so a leader
returning a node-list near _MAX_NODE_COUNT (10_000) cannot pin the loop while
the per-node policy callable runs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import _PROBE_TASK_CREATE_YIELD_EVERY, ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_cluster_info_filter_yields_on_large_reply() -> None:
    """A 2000-node accept-all response yields roughly N/K cooperative sleep(0)s."""
    from dqlitewire import NodeRole
    from dqlitewire.messages.responses import NodeInfo as _WireNodeInfo

    fake_nodes = [
        _WireNodeInfo(
            node_id=i + 1, address=f"10.0.0.{i // 256}.{i % 256}:9001", role=NodeRole.VOTER
        )
        for i in range(2000)
    ]

    cluster = ClusterClient(
        MemoryNodeStore(["10.0.0.0.0:9001"]),
        timeout=5.0,
    )

    class _FakeProto:
        async def cluster(self) -> list[_WireNodeInfo]:
            return fake_nodes

        async def get_leader(self) -> tuple[int, str]:
            return (1, "10.0.0.0.0:9001")

    class _FakeAdminCM:
        async def __aenter__(self) -> _FakeProto:
            return _FakeProto()

        async def __aexit__(self, *args: object) -> None:
            return None

    sleep_zero_calls = 0
    original_sleep = asyncio.sleep

    async def counting_sleep(delay: float, *args: object, **kwargs: object) -> None:
        nonlocal sleep_zero_calls
        if delay == 0:
            sleep_zero_calls += 1
        await original_sleep(delay, *args, **kwargs)

    async def fake_find_leader(*args: object, **kwargs: object) -> str:
        return "10.0.0.0.0:9001"

    # Explicit accept-all so the loop runs (effective_policy is None would skip it).
    accept_all: object = lambda _addr: True  # noqa: E731

    with (
        patch.object(cluster, "find_leader", AsyncMock(side_effect=fake_find_leader)),
        patch.object(cluster, "open_admin_connection", lambda _addr: _FakeAdminCM()),
        patch("dqliteclient.cluster.asyncio.sleep", new=counting_sleep),
    ):
        result = await cluster.cluster_info(policy=accept_all)  # type: ignore[arg-type]

    assert len(result) == 2000
    expected_min = (2000 // _PROBE_TASK_CREATE_YIELD_EVERY) - 1
    assert sleep_zero_calls >= expected_min, (
        f"cluster_info filter made only {sleep_zero_calls} sleep(0) calls; "
        f"expected at least {expected_min} for N=2000 nodes / "
        f"yield-every={_PROBE_TASK_CREATE_YIELD_EVERY}"
    )
