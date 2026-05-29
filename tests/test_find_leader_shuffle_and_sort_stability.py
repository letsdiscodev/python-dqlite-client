"""``find_leader``'s stampede avoidance needs the node shuffle to randomise
probe order across callers and the role-bucket sort to be stable so
equal-role nodes keep their shuffled order within a bucket."""

from __future__ import annotations

from dqliteclient.node_store import NodeInfo
from dqlitewire import NodeRole


def test_role_bucket_sort_is_stable_within_same_role() -> None:
    """The role-int sort is stable, preserving same-role nodes' relative
    order so a non-stable regression would lose shuffle avoidance."""
    nodes_in_input_order = [
        NodeInfo(1, "sb_a:9001", NodeRole.STANDBY),
        NodeInfo(2, "sb_b:9001", NodeRole.STANDBY),
        NodeInfo(3, "voter:9001", NodeRole.VOTER),
        NodeInfo(4, "sb_c:9001", NodeRole.STANDBY),
    ]
    out = sorted(nodes_in_input_order, key=lambda n: int(n.role))
    # VOTER first, then STANDBYs in input order (stable sort).
    assert [n.address for n in out] == [
        "voter:9001",
        "sb_a:9001",
        "sb_b:9001",
        "sb_c:9001",
    ]


def test_shuffle_then_sort_preserves_bucket_grouping() -> None:
    """Inspection pin: ``shuffle`` precedes ``sort`` in find_leader."""
    import inspect

    from dqliteclient.cluster import ClusterClient

    src = inspect.getsource(ClusterClient.find_leader)
    # The shuffle/sort pair lives in the _find_leader_impl helper.
    impl_src = inspect.getsource(ClusterClient._find_leader_impl)

    combined = src + impl_src
    shuffle_idx = combined.find("_cluster_random.shuffle")
    sort_idx = combined.find("nodes.sort")
    assert shuffle_idx >= 0, "Stampede-avoidance shuffle must be present"
    assert sort_idx >= 0, "Role-bucket sort must be present"
    assert shuffle_idx < sort_idx, (
        "Shuffle must precede the stable sort so equal-role nodes "
        "stay in shuffled order within each bucket"
    )
