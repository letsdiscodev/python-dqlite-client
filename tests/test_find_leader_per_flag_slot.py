"""Concurrent find_leader callers with different trust_server_heartbeat
values must not collapse onto one task, or the first caller's flag is
silently applied to both."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_concurrent_callers_with_different_flags_do_not_collapse() -> None:
    """Concurrent callers with True/False flags must each see their own
    flag honoured by the probe sweep."""
    store = MemoryNodeStore(["127.0.0.1:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    observed_flags: list[bool] = []

    async def fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        observed_flags.append(trust_server_heartbeat)
        await asyncio.sleep(0.01)  # yield so concurrent callers start
        return "127.0.0.1:9001"

    cluster._find_leader_impl = fake_impl

    a, b = await asyncio.gather(
        cluster.find_leader(trust_server_heartbeat=True),
        cluster.find_leader(trust_server_heartbeat=False),
    )

    assert a == "127.0.0.1:9001"
    assert b == "127.0.0.1:9001"
    assert sorted(observed_flags) == [False, True]


@pytest.mark.asyncio
async def test_concurrent_callers_with_same_flag_still_collapse() -> None:
    """Same-flag callers must still single-flight onto one task."""
    store = MemoryNodeStore(["127.0.0.1:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    call_count = 0

    async def fake_impl(*, trust_server_heartbeat: bool, policy: object = None) -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return "127.0.0.1:9001"

    cluster._find_leader_impl = fake_impl

    await asyncio.gather(
        cluster.find_leader(trust_server_heartbeat=False),
        cluster.find_leader(trust_server_heartbeat=False),
        cluster.find_leader(trust_server_heartbeat=False),
    )

    assert call_count == 1, f"Expected single-flight, got {call_count} calls"
