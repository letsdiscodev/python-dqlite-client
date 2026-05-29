"""``ClusterClient.transfer_leadership`` does NOT pre-invalidate the leader
cache (avoids a full N-node sweep on every call; find_leader's fast-path
falls through to the sweep on staleness). The finally: invalidation still
fires post-RPC. Matches go-dqlite's ``Client.Transfer``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_transfer_leadership_does_not_pre_invalidate_cache() -> None:
    """The cache is preserved through the find_leader entry."""
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
    """The finally: invalidation fires even when open_admin_connection
    raises, leaving the cache empty so the next call re-probes."""
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
    """The finally: invalidation also fires on success — the leader just
    stepped down, so the cached peer is suspect for the next call."""
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

    assert cc._get_last_known_leader() is None
