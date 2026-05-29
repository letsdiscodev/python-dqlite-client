"""Read-only admin RPCs preserve the leader cache on success, invalidate it on failure.

Membership-changing RPCs invalidate unconditionally because they can trigger elections.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    DqliteConnectionError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster(*, warm_cache: str = "warm:9001") -> ClusterClient:
    cc = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    cc._set_last_known_leader(warm_cache)
    cc.find_leader = AsyncMock(return_value="127.0.0.1:9001")
    return cc


def _make_admin_cm(*, raises: BaseException | None = None) -> MagicMock:
    fake_proto = MagicMock()
    if raises is None:
        fake_proto.cluster = AsyncMock(return_value=[])
        fake_proto.describe = AsyncMock(return_value=MagicMock(failure_domain=0, weight=0))
        fake_proto.weight = AsyncMock(return_value=None)
        fake_proto.dump = AsyncMock(return_value={})
        fake_proto.get_leader = AsyncMock(return_value=(1, "127.0.0.1:9001"))
    else:
        fake_proto.cluster = AsyncMock(side_effect=raises)
        fake_proto.describe = AsyncMock(side_effect=raises)
        fake_proto.weight = AsyncMock(side_effect=raises)
        fake_proto.dump = AsyncMock(side_effect=raises)
        fake_proto.get_leader = AsyncMock(side_effect=raises)
    fake_cm = MagicMock()
    fake_cm.__aenter__ = AsyncMock(return_value=fake_proto)
    fake_cm.__aexit__ = AsyncMock(return_value=None)
    return fake_cm


@pytest.mark.asyncio
async def test_cluster_info_success_preserves_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(return_value=_make_admin_cm())
    await cc.cluster_info()
    assert cc._last_known_leader == "warm:9001"


@pytest.mark.asyncio
async def test_describe_success_preserves_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(return_value=_make_admin_cm())
    await cc.describe()
    assert cc._last_known_leader == "warm:9001"


@pytest.mark.asyncio
async def test_set_weight_success_preserves_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(return_value=_make_admin_cm())
    await cc.set_weight(7)
    assert cc._last_known_leader == "warm:9001"


@pytest.mark.asyncio
async def test_dump_success_preserves_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(return_value=_make_admin_cm())
    await cc.dump("db")
    assert cc._last_known_leader == "warm:9001"


@pytest.mark.asyncio
async def test_cluster_info_failure_invalidates_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(
        return_value=_make_admin_cm(raises=DqliteConnectionError("flap"))
    )
    with pytest.raises(DqliteConnectionError):
        await cc.cluster_info()
    assert cc._last_known_leader is None


@pytest.mark.asyncio
async def test_describe_failure_invalidates_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(
        return_value=_make_admin_cm(raises=OperationalError("step-down", code=1))
    )
    with pytest.raises(OperationalError):
        await cc.describe()
    assert cc._last_known_leader is None


@pytest.mark.asyncio
async def test_dump_failure_invalidates_cache() -> None:
    cc = _make_cluster()
    cc.open_admin_connection = MagicMock(
        return_value=_make_admin_cm(raises=ProtocolError("garbled"))
    )
    with pytest.raises(ProtocolError):
        await cc.dump("db")
    assert cc._last_known_leader is None
