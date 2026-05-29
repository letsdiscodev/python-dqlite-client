"""``_probe_one`` releases the concurrency semaphore BEFORE the
``_verify_redirect`` re-probe, so redirect re-verifies are not serialized
behind the ``concurrent_leader_conns`` slots."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_verify_redirect_runs_outside_semaphore_slot() -> None:
    """Semaphore (size 1) must not be held during _verify_redirect: verifies
    must overlap."""
    cluster = ClusterClient(
        MemoryNodeStore(["a:9001", "b:9001", "c:9001"]),
        timeout=5.0,
        concurrent_leader_conns=1,
    )

    redirect_targets = {"a:9001": "ha:9001", "b:9001": "hb:9001", "c:9001": "hc:9001"}

    async def _query_leader_redirect(addr: str, **_kw: Any) -> str:
        return redirect_targets[addr]

    in_verify = 0
    max_concurrent_verify = 0
    verify_started = asyncio.Event()

    async def _slow_verify(addr: str, **_kw: Any) -> str:
        nonlocal in_verify, max_concurrent_verify
        in_verify += 1
        max_concurrent_verify = max(max_concurrent_verify, in_verify)
        verify_started.set()
        try:
            await asyncio.sleep(0.1)
            return addr
        finally:
            in_verify -= 1

    with (
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_query_leader_redirect)),
        patch.object(cluster, "_verify_redirect", AsyncMock(side_effect=_slow_verify)),
    ):
        await cluster.find_leader()

    assert max_concurrent_verify >= 2, (
        f"verify max-concurrent {max_concurrent_verify} indicates the "
        f"semaphore is still held across verify"
    )
