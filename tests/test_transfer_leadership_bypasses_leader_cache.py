"""``ClusterClient.transfer_leadership`` discards the cached leader
address BEFORE calling ``find_leader`` so the cache fast path does not
probe the very peer being asked to step down. After a recent successful
transfer the cache points at the just-stepped-down ex-leader; without
the pre-call invalidation, a chaos-monkey loop of transfers pays one
wasted cache-probe RTT per iteration.

Sibling admin RPCs (cluster_info, leader_info, describe, set_weight,
etc.) keep the cache fast path — they have no leader-step-down
semantic that contradicts the cache.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_transfer_leadership_clears_leader_cache_before_find_leader() -> None:
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    # Pre-seed the cache with a peer we want to verify gets discarded
    # BEFORE ``find_leader`` runs.
    cc._set_last_known_leader("stale.example.com:9001")
    observed_cache_at_find_leader: list[str | None] = []

    async def fake_find_leader(*_a: object, **_kw: object) -> str:
        observed_cache_at_find_leader.append(cc._get_last_known_leader())
        return "127.0.0.1:9001"

    cc.find_leader = fake_find_leader

    fake_proto = MagicMock()
    fake_proto.transfer = AsyncMock(return_value=None)
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    await cc.transfer_leadership(42)

    assert observed_cache_at_find_leader == [None], (
        f"transfer_leadership must invalidate the leader cache before "
        f"calling find_leader; cache at find_leader call was "
        f"{observed_cache_at_find_leader!r}"
    )


@pytest.mark.asyncio
async def test_transfer_leadership_still_invalidates_cache_after_failure() -> None:
    """The pre-call invalidation must not regress the existing
    post-call invalidation that handles leader-flip-mid-RPC: a failure
    inside ``open_admin_connection`` must still leave the cache empty
    on the way out."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader("stale.example.com:9001")

    cc.find_leader = AsyncMock(return_value="127.0.0.1:9001")

    # Mid-RPC failure: the admin RPC raises after the cache was already
    # cleared pre-call. After the finally: the cache must still be None
    # (re-clearing is a no-op for the success path).
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(side_effect=ConnectionResetError("boom"))
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    # Manually mutate the cache mid-flight to simulate a leader-flip
    # racing with the transfer call.
    async def find_leader_then_repoison(*_a: object, **_kw: object) -> str:
        # A concurrent caller has already discovered a fresh leader
        # while transfer_leadership is in flight.
        cc._set_last_known_leader("freshly-discovered:9001")
        return "127.0.0.1:9001"

    cc.find_leader = find_leader_then_repoison

    with pytest.raises(ConnectionResetError):
        await cc.transfer_leadership(42)

    assert cc._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_transfer_leadership_loop_does_not_burn_cache_probes() -> None:
    """Smoke test: a tight loop of transfer_leadership calls observes
    ``None`` cache at every ``find_leader`` entry — confirming the cache
    fast path is consistently skipped."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    observed: list[str | None] = []

    async def fake_find_leader(*_a: object, **_kw: object) -> str:
        observed.append(cc._get_last_known_leader())
        # Simulate a successful find_leader that re-populates the cache
        # so the next iteration would otherwise see it set.
        cc._set_last_known_leader("127.0.0.1:9001")
        return "127.0.0.1:9001"

    cc.find_leader = fake_find_leader

    fake_proto = MagicMock()
    fake_proto.transfer = AsyncMock(return_value=None)
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    for _ in range(5):
        await cc.transfer_leadership(42)
        await asyncio.sleep(0)

    assert observed == [None] * 5
