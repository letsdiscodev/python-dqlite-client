"""``LeaderInfo`` / ``NodeMetadata`` (returned by public methods) must be top-level imports."""

import dqliteclient
from dqliteclient.cluster import LeaderInfo as ClusterLeaderInfo
from dqliteclient.cluster import NodeMetadata as ClusterNodeMetadata


def test_leader_info_is_top_level_export() -> None:
    assert "LeaderInfo" in dqliteclient.__all__
    assert dqliteclient.LeaderInfo is ClusterLeaderInfo


def test_node_metadata_is_top_level_export() -> None:
    assert "NodeMetadata" in dqliteclient.__all__
    assert dqliteclient.NodeMetadata is ClusterNodeMetadata
