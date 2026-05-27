"""Pin: ``ClusterClient.cluster_info``'s redirect-policy filter
yields periodically so a hostile or buggy leader returning a
node-list close to ``_MAX_NODE_COUNT`` (10_000) cannot pin the
loop while the policy callable runs against each entry.

The default policy calls ``parse_address`` + ``ipaddress.ip_address``
per node — ~tens of microseconds per call on commodity hardware,
hundreds of milliseconds total at the cap. Yield every K so the
loop stays responsive.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import _PROBE_TASK_CREATE_YIELD_EVERY, ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_cluster_info_filter_yields_on_large_reply() -> None:
    """A ``cluster_info`` response with N=2000 nodes that all match
    the policy must result in N / K + cooperative ``sleep(0)``
    calls during the filter loop."""
    # Build a leader that returns 2000 nodes; policy rejects none
    # so we exercise the happy path through the loop body.
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

    # Stub the wire path so cluster() returns our 2000-node payload
    # without doing real network IO. Patch open_admin_connection
    # to yield a fake protocol whose .cluster() returns the list.

    class _FakeProto:
        async def cluster(self) -> list[_WireNodeInfo]:
            return fake_nodes

        async def get_leader(self) -> tuple[int, str]:
            # Return the seed address — no redirect needed.
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

    # Pass an explicit accept-all policy so the filter loop runs
    # (the early-exit on ``effective_policy is None`` would skip
    # the loop entirely).
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
