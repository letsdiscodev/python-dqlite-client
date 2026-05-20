"""``leader_info(policy=...)`` re-validates the responder's
self-reported leader address against a redirect policy when
leadership flipped between ``find_leader`` and the follow-up
``protocol.get_leader()`` round-trip.

Without this gate, a hostile follower reached on the second hop
could return an attacker-controlled address as "leader" and the
caller would receive that address verbatim — bypassing every check
the instance-level ``redirect_policy`` was designed to enforce.

The check is skipped when the responder confirms itself (the same
address ``find_leader`` already approved through its own
``_check_redirect`` arm). This matches the precedence used by
:meth:`cluster_info` and the find-leader probe path.
"""

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
    """Cluster whose ``find_leader`` returns ``leader:9001`` and whose
    admin protocol's ``get_leader()`` returns the configured tuple.

    When the responder reports a flipped address (``responder_address``
    != ``"leader:9001"``), ``leader_info`` re-probes via
    ``_verify_redirect`` and then re-asks ``get_leader`` on the
    verified target. ``verified_node_id`` / ``verified_address``
    drive the verified responder's get_leader return; by default the
    re-probe returns the responder_address unchanged."""
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        redirect_policy=redirect_policy,
    )
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    # Mock the re-probe to succeed against the responder's address —
    # this is the "verified" target. Returning a non-None mirrors the
    # production path where the hint self-confirmed.
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
        # The first call (to leader_addr) returns fake_proto; the
        # second call (to the verified target after the re-probe)
        # returns verified_proto.
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
    """Without ``policy=`` and without an instance-level
    ``redirect_policy``, the responder's address passes through
    unchecked (preserves prior callers)."""
    cluster = _build_cluster_with_responder(7, "responder:9001")
    info = await cluster.leader_info()
    assert info is not None
    assert info.node_id == 7
    assert info.address == "responder:9001"


@pytest.mark.asyncio
async def test_no_flip_skips_policy_check() -> None:
    """When the responder confirms itself (same address as
    ``find_leader`` returned), the policy is NOT consulted — the
    responder was already approved by ``find_leader``'s own
    ``_check_redirect`` arm."""

    def reject_everything(_addr: str) -> bool:
        return False

    cluster = _build_cluster_with_responder(7, "leader:9001")
    info = await cluster.leader_info(policy=reject_everything)
    assert info is not None
    assert info.address == "leader:9001"


@pytest.mark.asyncio
async def test_per_call_policy_rejects_flipped_address() -> None:
    """``policy=...`` rejects a flipped responder's address even when
    the instance-level ``redirect_policy`` would have admitted it."""
    cluster = _build_cluster_with_responder(7, "10.99.99.99:9001")

    def reject_10_dot(addr: str) -> bool:
        host, _ = addr.rsplit(":", 1)
        return not host.startswith("10.")

    with pytest.raises(ClusterPolicyError, match="redirect.*rejected"):
        await cluster.leader_info(policy=reject_10_dot)


@pytest.mark.asyncio
async def test_instance_redirect_policy_used_when_no_per_call_policy() -> None:
    """When no per-call ``policy`` is provided, the instance-level
    ``redirect_policy`` is consulted on a flipped address."""

    def reject_10_dot(addr: str) -> bool:
        host, _ = addr.rsplit(":", 1)
        return not host.startswith("10.")

    cluster = _build_cluster_with_responder(7, "10.99.99.99:9001", redirect_policy=reject_10_dot)

    with pytest.raises(ClusterPolicyError, match="redirect.*rejected"):
        await cluster.leader_info()


@pytest.mark.asyncio
async def test_no_leader_response_returns_none() -> None:
    """When the responder reports ``(node_id=0, address="")`` (mid-
    election sentinel), ``leader_info`` returns ``None`` and never
    invokes the policy."""
    cluster = _build_cluster_with_responder(0, "")
    info = await cluster.leader_info(policy=lambda _addr: False)
    assert info is None
