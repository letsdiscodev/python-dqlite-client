"""dqliteclient.cluster exposes no NodeInfo attribute: the wire NodeInfo is
imported as _WireNodeInfo so it cannot silently shadow the distinct user-facing
dqliteclient.NodeInfo (node_store) for callers importing from cluster.
"""


def test_cluster_module_does_not_expose_node_info_attribute() -> None:
    import dqliteclient.cluster as cluster_mod

    assert not hasattr(cluster_mod, "NodeInfo"), (
        "dqliteclient.cluster.NodeInfo was the wire NodeInfo silently "
        "shadowing dqliteclient.node_store.NodeInfo (the user-facing "
        "class). Use _WireNodeInfo at the wire-import site so the two "
        "are explicitly distinguishable."
    )


def test_wire_alias_resolves_to_wire_node_info_class() -> None:
    import dqliteclient.cluster as cluster_mod
    import dqlitewire.messages.responses as wire_responses

    assert cluster_mod._WireNodeInfo is wire_responses.NodeInfo  # type: ignore[attr-defined]


def test_user_facing_node_info_is_node_store_node_info() -> None:
    import dqliteclient
    import dqliteclient.node_store

    assert dqliteclient.NodeInfo is dqliteclient.node_store.NodeInfo


def test_two_node_info_classes_are_distinct() -> None:
    import dqliteclient
    import dqlitewire.messages.responses

    assert dqliteclient.NodeInfo is not dqlitewire.messages.responses.NodeInfo, (
        "These are deliberately distinct classes — the cluster module "
        "now uses the wire class only as _WireNodeInfo to avoid silent "
        "shadowing of the user-facing class."
    )
