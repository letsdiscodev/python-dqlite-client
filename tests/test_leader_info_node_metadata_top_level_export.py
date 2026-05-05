"""``LeaderInfo`` and ``NodeMetadata`` are returned by public
``ClusterClient.leader_info()`` / ``ClusterClient.describe()`` methods,
so they MUST be reachable via the package top-level import.

Before this fix, callers had to write
``from dqliteclient.cluster import LeaderInfo`` — branching the public
surface across two import paths.
"""

import dqliteclient
from dqliteclient.cluster import LeaderInfo as ClusterLeaderInfo
from dqliteclient.cluster import NodeMetadata as ClusterNodeMetadata


def test_leader_info_is_top_level_export() -> None:
    assert "LeaderInfo" in dqliteclient.__all__
    assert dqliteclient.LeaderInfo is ClusterLeaderInfo


def test_node_metadata_is_top_level_export() -> None:
    assert "NodeMetadata" in dqliteclient.__all__
    assert dqliteclient.NodeMetadata is ClusterNodeMetadata
