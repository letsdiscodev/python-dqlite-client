"""``ClusterClient.transfer_leadership`` does NOT pre-invalidate the
leader cache before calling ``find_leader``. The find_leader fast-
path handles the "cache points at a stepped-down peer" case by
falling through to the sweep on a leader-flip code — one extra probe
RTT, not a full sweep.

Pre-invalidating would force a full N-node sweep on every transfer
call, including the warm-cached no-op case (transfer to the same
node, or transfer the cluster rejects). Matches go-dqlite's
``Client.Transfer`` which doesn't pre-invalidate the leader tracker.

The post-RPC invalidation in the ``finally:`` still fires (covers
the leader-step-down-mid-RPC case).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_transfer_leadership_does_not_pre_invalidate_cache() -> None:
    """The cache is preserved through the find_leader entry — the
    fast-path handles staleness via fall-through-to-sweep."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader("warm.example.com:9001")
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

    assert observed_cache_at_find_leader == ["warm.example.com:9001"], (
        f"transfer_leadership must NOT pre-invalidate the cache; the "
        f"find_leader fast-path handles stepped-down peers via the "
        f"sweep fallback. Got cache history: {observed_cache_at_find_leader!r}"
    )


@pytest.mark.asyncio
async def test_transfer_leadership_still_invalidates_cache_after_failure() -> None:
    """The post-RPC finally invalidation still fires: a failure inside
    ``open_admin_connection`` must leave the cache empty on the way
    out so the next call re-probes."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader("warm.example.com:9001")

    cc.find_leader = AsyncMock(return_value="127.0.0.1:9001")

    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(side_effect=ConnectionResetError("boom"))
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    with pytest.raises(ConnectionResetError):
        await cc.transfer_leadership(42)

    assert cc._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_transfer_leadership_invalidates_cache_on_success() -> None:
    """The post-RPC finally invalidation also fires on success: the
    transfer succeeded, so the leader-just-stepped-down semantic
    means the cached peer is suspect for the next call."""
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader("warm.example.com:9001")

    cc.find_leader = AsyncMock(return_value="127.0.0.1:9001")

    fake_proto = MagicMock()
    fake_proto.transfer = AsyncMock(return_value=None)
    fake_admin_cm = MagicMock()
    fake_admin_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_admin_cm.__aexit__ = AsyncMock(return_value=None)
    cc.open_admin_connection = MagicMock(return_value=fake_admin_cm)

    await cc.transfer_leadership(42)

    # Post-RPC invalidation discipline: the leader just stepped down,
    # so the cache (which may have been updated by find_leader during
    # the transfer's own sweep) is suspect for the next call.
    assert cc._get_last_known_leader() is None
