"""``cluster_info`` tolerates the (node_id != 0, address == "") RAFT_NOMEM
transient on its post-find_leader get_leader(), treating it like the canonical
(0, "") sentinel rather than chasing the empty address into a false ClusterError.
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
    """(N, "") RAFT_NOMEM transient: read config from the current responder."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(99, ""))  # RAFT_NOMEM: id, no address
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    result = await cluster.cluster_info()

    assert result == nodes
    proto.get_leader.assert_awaited_once_with()
    proto.cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_no_leader_known_sentinel_falls_back_to_local_responder() -> None:
    """Canonical (0, "") sentinel also falls back to the local responder."""
    cluster = ClusterClient(MemoryNodeStore(["leader:9001"]), timeout=2.0)
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    nodes = [NodeInfo(node_id=1, address="leader:9001", role=NodeRole.VOTER)]
    proto = MagicMock()
    proto.get_leader = AsyncMock(return_value=(0, ""))
    proto.cluster = AsyncMock(return_value=nodes)
    cluster.open_admin_connection = MagicMock(return_value=_make_admin_cm(proto))

    result = await cluster.cluster_info()

    assert result == nodes
