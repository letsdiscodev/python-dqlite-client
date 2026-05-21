"""Pin: ``cluster_info`` re-confirms leadership on the open admin
connection BEFORE reading the cluster configuration, mirroring the
leader-flip re-verify arm in ``leader_info``.

Without this arm, a mid-RPC leader flip between ``find_leader`` and
the ``protocol.cluster()`` round-trip can land on a stepped-down
responder whose ``cluster`` reply lists a node not in the operator's
``redirect_policy`` allowlist. The returned-nodes filter would
silently drop the actual current leader, breaking subsequent
``find_leader`` calls that consume the result via ``set_nodes(...)``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _make_admin_cm(proto: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=proto)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_cluster_info_reconfirm_no_flip_calls_get_leader_once() -> None:
    """No-flip happy path: ``get_leader`` returns the same address as
    ``find_leader`` did, so the cluster config is read from the
    already-open connection without a second admin connection."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(1, "leader:9001"))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    result = await cluster.cluster_info()

    assert result == nodes
    proto.get_leader.assert_awaited_once_with()
    proto.cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cluster_info_leader_flip_mid_rpc_re_verifies_and_re_reads() -> None:
    """Mid-flip path: ``get_leader`` hands back a different address,
    triggering ``_verify_redirect`` and a fresh admin connection
    against the verified leader. The cluster RPC then lands on the
    verified leader, not the original (now-stale) responder."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._check_redirect = MagicMock(return_value=None)
    cluster._verify_redirect = AsyncMock(return_value="new-leader:9001")

    nodes_from_stale = [
        NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER),
        # The stale responder would still return its own view; the
        # test asserts we DON'T read from it.
    ]
    nodes_from_verified = [
        NodeInfo(node_id=1, address="leader:9001", role=NodeRole.SPARE),
        NodeInfo(node_id=2, address="new-leader:9001", role=NodeRole.VOTER),
    ]

    stale_proto = MagicMock()
    stale_proto.get_leader = AsyncMock(return_value=(2, "new-leader:9001"))
    stale_proto.cluster = AsyncMock(return_value=nodes_from_stale)

    verified_proto = MagicMock()
    verified_proto.get_leader = AsyncMock(return_value=(2, "new-leader:9001"))
    verified_proto.cluster = AsyncMock(return_value=nodes_from_verified)

    admin_cms = [_make_admin_cm(stale_proto), _make_admin_cm(verified_proto)]
    cluster.open_admin_connection = MagicMock(side_effect=admin_cms)

    result = await cluster.cluster_info()

    # Verified leader's view is the one returned (not the stale view).
    assert result == nodes_from_verified
    # _verify_redirect must run with the responder's new-leader hint.
    cluster._verify_redirect.assert_awaited_once()
    args, kwargs = cluster._verify_redirect.call_args
    assert (args and args[0] == "new-leader:9001") or kwargs.get("address") == "new-leader:9001"
    # The stale responder's cluster() is NEVER called — we re-route to
    # the verified leader before reading.
    stale_proto.cluster.assert_not_called()
    verified_proto.cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cluster_info_leader_flip_unverifiable_raises_cluster_error() -> None:
    """When ``_verify_redirect`` returns ``None`` for the responder's
    hinted new leader, ``cluster_info`` MUST raise ``ClusterError``
    rather than silently returning the stale view."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")
    cluster._set_last_known_leader("leader:9001")
    cluster._check_redirect = MagicMock(return_value=None)
    cluster._verify_redirect = AsyncMock(return_value=None)

    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(2, "ghost-leader:9001"))
    proto.cluster = AsyncMock(return_value=[])
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    with pytest.raises(ClusterError, match="leadership flipped"):
        await cluster.cluster_info()
    # Cache invalidated so the next find_leader runs a full sweep.
    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_cluster_info_no_leader_known_response_treated_as_no_flip() -> None:
    """A ``(node_id=0, address="")`` response from ``get_leader`` is the
    server's "no leader yet" reply during an election; this is not a
    flip but a no-leader signal. The existing reading path proceeds
    (the upstream RPC will surface the appropriate failure)."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(0, ""))
    proto.cluster = AsyncMock(return_value=[])
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    # No exception, no extra admin connection.
    result = await cluster.cluster_info()
    assert result == []
    proto.cluster.assert_awaited_once_with()
