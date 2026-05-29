"""``leader_info(policy=...)`` re-validates a flipped responder's self-reported leader
against a redirect policy; without it a hostile follower could smuggle an attacker address.
Skipped when the responder confirms itself (already approved by ``find_leader``)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def _build_cluster_with_responder(
    responder_node_id: int = 7,
    responder_address: str = "leader:9001",
    *,
    redirect_policy=None,
    verified_node_id: int | None = None,
    verified_address: str | None = None,
) -> ClusterClient:
    """Cluster whose ``find_leader`` returns ``leader:9001``; on a flipped responder it
    re-probes via ``_verify_redirect`` then re-asks ``get_leader`` on the verified target."""
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        redirect_policy=redirect_policy,
    )
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._verify_redirect = AsyncMock(return_value=responder_address)

    fake_proto = MagicMock()
    fake_proto.get_leader = AsyncMock(return_value=(responder_node_id, responder_address))

    verified_proto = MagicMock()
    verified_proto.get_leader = AsyncMock(
        return_value=(
            verified_node_id if verified_node_id is not None else responder_node_id,
            verified_address if verified_address is not None else responder_address,
        )
    )

    def admin_cm(target: str) -> MagicMock:
        cm = MagicMock()
        if target == responder_address and responder_address != "leader:9001":
            cm.__aenter__ = AsyncMock(return_value=verified_proto)
        else:
            cm.__aenter__ = AsyncMock(return_value=fake_proto)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    cluster.open_admin_connection = MagicMock(side_effect=admin_cm)

    return cluster


@pytest.mark.asyncio
async def test_no_policy_returns_unchecked() -> None:
    """With no policy at all, the responder's address passes through unchecked."""
    cluster = _build_cluster_with_responder(7, "responder:9001")
    info = await cluster.leader_info()
    assert info is not None
    assert info.node_id == 7
    assert info.address == "responder:9001"


@pytest.mark.asyncio
async def test_no_flip_skips_policy_check() -> None:
    """A self-confirming responder skips the policy — already approved by ``find_leader``."""

    def reject_everything(_addr: str) -> bool:
        return False

    cluster = _build_cluster_with_responder(7, "leader:9001")
    info = await cluster.leader_info(policy=reject_everything)
    assert info is not None
    assert info.address == "leader:9001"


@pytest.mark.asyncio
async def test_per_call_policy_rejects_flipped_address() -> None:
    """Per-call ``policy=`` overrides the instance ``redirect_policy`` for a flipped address."""
    cluster = _build_cluster_with_responder(7, "10.99.99.99:9001")

    def reject_10_dot(addr: str) -> bool:
        host, _ = addr.rsplit(":", 1)
        return not host.startswith("10.")

    with pytest.raises(ClusterPolicyError, match="redirect.*rejected"):
        await cluster.leader_info(policy=reject_10_dot)


@pytest.mark.asyncio
async def test_instance_redirect_policy_used_when_no_per_call_policy() -> None:
    """Without a per-call policy, the instance ``redirect_policy`` is consulted on a flip."""

    def reject_10_dot(addr: str) -> bool:
        host, _ = addr.rsplit(":", 1)
        return not host.startswith("10.")

    cluster = _build_cluster_with_responder(7, "10.99.99.99:9001", redirect_policy=reject_10_dot)

    with pytest.raises(ClusterPolicyError, match="redirect.*rejected"):
        await cluster.leader_info()


@pytest.mark.asyncio
async def test_third_hop_vaddress_filtered_by_policy() -> None:
    """The verified responder's reported leader (third hop) must be policy-checked too:
    a compromised allowlisted follower could otherwise return an attacker address."""

    def reject_outside_10_subnet(addr: str) -> bool:
        host, _ = addr.rsplit(":", 1)
        return host.startswith("10.")

    cluster = _build_cluster_with_responder(
        responder_node_id=7,
        responder_address="10.0.0.5:9001",
        verified_node_id=9,
        verified_address="192.168.1.1:9001",
    )

    with pytest.raises(ClusterPolicyError, match="redirect.*rejected"):
        await cluster.leader_info(policy=reject_outside_10_subnet)


@pytest.mark.asyncio
async def test_third_hop_vaddress_passes_when_within_policy() -> None:
    """Positive twin: a third hop within the policy round-trips cleanly."""

    def reject_outside_10_subnet(addr: str) -> bool:
        host, _ = addr.rsplit(":", 1)
        return host.startswith("10.")

    cluster = _build_cluster_with_responder(
        responder_node_id=7,
        responder_address="10.0.0.5:9001",
        verified_node_id=9,
        verified_address="10.0.0.9:9001",
    )

    info = await cluster.leader_info(policy=reject_outside_10_subnet)
    assert info is not None
    assert info.node_id == 9
    assert info.address == "10.0.0.9:9001"


@pytest.mark.asyncio
async def test_no_leader_response_returns_none() -> None:
    """The ``(0, "")`` mid-election sentinel returns None and never invokes the policy."""
    cluster = _build_cluster_with_responder(0, "")
    info = await cluster.leader_info(policy=lambda _addr: False)
    assert info is None
