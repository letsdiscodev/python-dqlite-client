"""Pin: ``_probe_one`` in the parallel sweep releases the
concurrency semaphore BEFORE the redirect re-probe via
``_verify_redirect``. Without this discipline, a redirect-stampede
plus leader-flip would serialize at
``concurrent_leader_conns`` slots × (attempt_timeout + verify +
drain), bottlenecking exactly the case the parallel sweep was meant
to amortize.

The trade-off documented in the issue: ``concurrent_leader_conns``
bounds initial probe dials but not redirect re-verify dials. Under
a full-cluster redirect stampede, the verify fan-out can briefly
exceed the slot count.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_verify_redirect_runs_outside_semaphore_slot() -> None:
    """Semaphore must NOT be held while _verify_redirect is
    in-flight. Drive the verify to a long sleep and observe that
    multiple verifies run in parallel even when the semaphore is
    sized to 1.
    """
    cluster = ClusterClient(
        MemoryNodeStore(["a:9001", "b:9001", "c:9001"]),
        timeout=5.0,
        concurrent_leader_conns=1,
    )

    # Each probe redirects to a unique address.
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

    # If the semaphore (size=1) had been held across the verify, only
    # one verify would have been in-flight at a time
    # (max_concurrent_verify == 1). The whole point of releasing the
    # slot between phase 1 and phase 2 is that verifies can overlap.
    assert max_concurrent_verify >= 2, (
        f"verify max-concurrent {max_concurrent_verify} indicates the "
        f"semaphore is still held across verify"
    )
