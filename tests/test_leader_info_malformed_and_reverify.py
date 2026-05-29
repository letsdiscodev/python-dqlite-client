"""Pin: ``leader_info`` raises ``ProtocolError`` for malformed ``(0, addr!="")`` but
tolerates the ``(id>0, "")`` RAFT_NOMEM transient (recvUpdateLeader) by surfacing None."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ProtocolError
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster(
    get_leader_return: tuple[int, str],
    *,
    verified_return: str | None = None,
    verified_get_leader: tuple[int, str] | None = None,
) -> ClusterClient:
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._verify_redirect = AsyncMock(return_value=verified_return)

    fake_proto = MagicMock()
    fake_proto.get_leader = AsyncMock(return_value=get_leader_return)

    verified_proto = MagicMock()
    if verified_get_leader is not None:
        verified_proto.get_leader = AsyncMock(return_value=verified_get_leader)

    def admin_cm(target: str) -> MagicMock:
        cm = MagicMock()
        if verified_return is not None and target == verified_return:
            cm.__aenter__ = AsyncMock(return_value=verified_proto)
        else:
            cm.__aenter__ = AsyncMock(return_value=fake_proto)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    cluster.open_admin_connection = MagicMock(side_effect=admin_cm)
    return cluster


@pytest.mark.asyncio
async def test_malformed_zero_id_nonempty_address_raises_protocol_error() -> None:
    cluster = _make_cluster((0, "attacker:9001"))
    with pytest.raises(ProtocolError, match="malformed"):
        await cluster.leader_info()


@pytest.mark.asyncio
async def test_nonzero_id_empty_address_returns_none() -> None:
    """``(7, "")`` is the RAFT_NOMEM transient — return None, not raise."""
    cluster = _make_cluster((7, ""))
    info = await cluster.leader_info()
    assert info is None


@pytest.mark.asyncio
async def test_zero_id_empty_address_returns_none() -> None:
    cluster = _make_cluster((0, ""))
    info = await cluster.leader_info()
    assert info is None


@pytest.mark.asyncio
async def test_flipped_address_reverified_via_verify_redirect() -> None:
    """On leader flip, uses the verified target's get_leader, not the stale hint's."""
    cluster = _make_cluster(
        (7, "flipped:9001"),
        verified_return="flipped:9001",
        verified_get_leader=(7, "flipped:9001"),
    )
    info = await cluster.leader_info()
    assert info is not None
    assert info.node_id == 7
    assert info.address == "flipped:9001"
    verify_mock = cluster._verify_redirect
    assert isinstance(verify_mock, AsyncMock)
    verify_mock.assert_awaited_once_with(
        "flipped:9001",
        trust_server_heartbeat=False,
    )


@pytest.mark.asyncio
async def test_flipped_address_verification_fails_returns_none() -> None:
    """When the re-probe returns None, surface None, not the suspect address."""
    cluster = _make_cluster(
        (7, "stale:9001"),
        verified_return=None,
    )
    info = await cluster.leader_info()
    assert info is None
    verify_mock = cluster._verify_redirect
    assert isinstance(verify_mock, AsyncMock)
    verify_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_leader_info_success_preserves_cache() -> None:
    """SUCCESS must not invalidate ``_last_known_leader`` — the responder just answered."""
    cluster = ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]), timeout=2.0)
    cluster._set_last_known_leader("warm:9001")
    cluster.find_leader = AsyncMock(return_value="127.0.0.1:9001")
    fake_proto = MagicMock()
    fake_proto.get_leader = AsyncMock(return_value=(1, "127.0.0.1:9001"))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=fake_proto)
    cm.__aexit__ = AsyncMock(return_value=None)
    cluster.open_admin_connection = MagicMock(return_value=cm)

    info = await cluster.leader_info()
    assert info is not None
    assert cluster._get_last_known_leader() == "warm:9001", (
        "leader_info SUCCESS must preserve the cache"
    )


@pytest.mark.asyncio
async def test_verified_target_returns_zero_empty_returns_none() -> None:
    """Inner re-validation: a verified target reporting ``(0, "")`` surfaces None."""
    cluster = _make_cluster(
        (7, "flipped:9001"),
        verified_return="flipped:9001",
        verified_get_leader=(0, ""),
    )
    info = await cluster.leader_info()
    assert info is None


@pytest.mark.asyncio
async def test_verified_target_returns_zero_nonempty_raises_protocol_error() -> None:
    """Inner re-validation: a verified target reporting ``(0, addr!="")`` must still raise,
    so an attacker-influenced stale hint can't smuggle a malformed shape past as None."""
    cluster = _make_cluster(
        (7, "flipped:9001"),
        verified_return="flipped:9001",
        verified_get_leader=(0, "attacker:9001"),
    )
    with pytest.raises(ProtocolError, match="malformed"):
        await cluster.leader_info()


@pytest.mark.asyncio
async def test_verified_target_returns_nonzero_empty_returns_none() -> None:
    """Inner re-validation: a verified target reporting ``(7, "")`` mirrors the outer arm."""
    cluster = _make_cluster(
        (7, "flipped:9001"),
        verified_return="flipped:9001",
        verified_get_leader=(7, ""),
    )
    info = await cluster.leader_info()
    assert info is None
