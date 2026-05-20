"""Pin: ``leader_info`` raises ``ProtocolError`` for malformed
``(node_id, address)`` shapes and re-probes via ``_verify_redirect``
when leadership flipped between ``find_leader`` and the follow-up
``get_leader`` round-trip.

The malformed-shape check mirrors the sibling ``_query_leader`` arm
which raises ``ProtocolError`` for the same shape (raft_leader is
atomic — both fields are set together or neither). The re-probe
mirrors the parallel-sweep ``_probe_one`` and fast-path arms which
already call ``_verify_redirect`` on the hint before trusting it.
"""

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
async def test_malformed_nonzero_id_empty_address_raises_protocol_error() -> None:
    cluster = _make_cluster((7, ""))
    with pytest.raises(ProtocolError, match="malformed"):
        await cluster.leader_info()


@pytest.mark.asyncio
async def test_zero_id_empty_address_returns_none() -> None:
    cluster = _make_cluster((0, ""))
    info = await cluster.leader_info()
    assert info is None


@pytest.mark.asyncio
async def test_flipped_address_reverified_via_verify_redirect() -> None:
    """On leader flip mid-call, ``leader_info`` calls ``_verify_redirect``
    and uses the verified target's get_leader output (not the stale
    hint's)."""
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
    """Stale-hint case: the re-probe returns None, so leader_info
    surfaces None rather than the suspect address."""
    cluster = _make_cluster(
        (7, "stale:9001"),
        verified_return=None,
    )
    info = await cluster.leader_info()
    assert info is None
    verify_mock = cluster._verify_redirect
    assert isinstance(verify_mock, AsyncMock)
    verify_mock.assert_awaited_once()
