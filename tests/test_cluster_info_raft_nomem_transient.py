"""Pin: ``cluster_info`` tolerates the ``(node_id != 0, address == "")``
RAFT_NOMEM transient on its post-``find_leader`` ``get_leader()``
round-trip, mirroring ``leader_info``'s handling of the same wire
shape.

The wire layer documents two distinct "leader not currently known"
shapes (see ``dqlitewire/messages/responses.py:441-470``):

1. ``(node_id == 0, address == "")`` — canonical sentinel.
2. ``(node_id != 0, address == "")`` — RAFT_NOMEM transient.

Before this fix ``cluster_info`` recognised only shape (1) and ran
the redirect-chase on shape (2). The chase dialled the empty address,
failed verification, and surfaced as ``ClusterError("leadership
flipped mid-RPC and the responder's hint did not re-confirm")`` —
operator-confusing misclassification of a recoverable transient.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _make_admin_cm(proto: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=proto)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_raft_nomem_transient_falls_back_to_local_responder() -> None:
    """`(node_id=N, address="")` from the responder is the RAFT_NOMEM
    transient. ``cluster_info`` reads cluster configuration from the
    current responder (the address ``find_leader`` already approved),
    matching the discipline applied to the ``(0, "")`` sentinel."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    # RAFT_NOMEM transient: known leader id, no address yet.
    proto.get_leader = AsyncMock(return_value=(99, ""))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    # No ClusterError — fall through to local-responder cluster view.
    result = await cluster.cluster_info()

    assert result == nodes
    proto.get_leader.assert_awaited_once_with()
    proto.cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_no_leader_known_sentinel_falls_back_to_local_responder() -> None:
    """Sibling positive: the canonical (0, "") sentinel was already
    handled correctly before the fix. Pin the behaviour so the fix
    doesn't accidentally regress the canonical path."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(0, ""))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    result = await cluster.cluster_info()

    assert result == nodes
