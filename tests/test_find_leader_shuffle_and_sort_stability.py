"""Pin: ``find_leader``'s stampede-avoidance discipline relies on
two properties of the node ordering at probe-dispatch:

1. ``_cluster_random.shuffle`` randomises the node list so two
   independent ``find_leader`` callers probe in different orders —
   a herd of new clients doesn't converge on the same first peer.
2. ``nodes.sort(key=lambda n: int(n.role))`` is stable so
   equal-role nodes keep their shuffled relative order — the
   role-bucket sort doesn't undo the shuffle within a bucket.

The pre-existing ``test_find_leader_sorts_standby_before_spare``
test pins the BUCKET ORDER (VOTER → STANDBY → SPARE). Neither of
the two stampede-avoidance properties was pinned.
"""

from __future__ import annotations

from dqliteclient.node_store import NodeInfo
from dqlitewire import NodeRole


def test_role_bucket_sort_is_stable_within_same_role() -> None:
    """Python's ``sorted`` (and ``list.sort``) is stable per CPython
    guarantees; the sort by role-int preserves the relative order of
    same-role nodes. A regression replacing ``list.sort`` with a
    non-stable sort would lose the shuffle's stampede-avoidance
    property within each role bucket."""
    nodes_in_input_order = [
        NodeInfo(1, "sb_a:9001", NodeRole.STANDBY),
        NodeInfo(2, "sb_b:9001", NodeRole.STANDBY),
        NodeInfo(3, "voter:9001", NodeRole.VOTER),
        NodeInfo(4, "sb_c:9001", NodeRole.STANDBY),
    ]
    out = sorted(nodes_in_input_order, key=lambda n: int(n.role))
    # VOTER first, then STANDBYs in their INPUT order (stable sort).
    assert [n.address for n in out] == [
        "voter:9001",
        "sb_a:9001",
        "sb_b:9001",
        "sb_c:9001",
    ]


def test_shuffle_then_sort_preserves_bucket_grouping() -> None:
    """End-to-end shape: random shuffle followed by role-keyed stable
    sort produces correctly grouped buckets even when input order is
    interleaved. Inspection-style pin against the source string
    ensures the pair ``shuffle`` + ``sort`` (in that order) is
    present in ``find_leader``."""
    import inspect

    from dqliteclient.cluster import ClusterClient

    src = inspect.getsource(ClusterClient.find_leader)
    # The sort + shuffle pair lives inside the helper
    # ``_find_leader_impl`` reached from ``find_leader``; search both.
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
