"""Pin: ``dqliteclient.cluster`` does NOT expose a ``NodeInfo``
attribute that silently shadows ``dqliteclient.NodeInfo`` (the
user-facing class).

There are two ``NodeInfo`` dataclasses in this package:
``dqlitewire.messages.responses.NodeInfo`` (wire-layer; used
internally to decode ``ServersResponse``) and
``dqliteclient.node_store.NodeInfo`` (user-facing; re-exported
as ``dqliteclient.NodeInfo``). They share field shape but are
distinct types. Before the rename, the cluster module imported
the wire class without an alias and exposed it as
``dqliteclient.cluster.NodeInfo``, silently shadowing
``dqliteclient.NodeInfo`` for any caller doing
``from dqliteclient.cluster import NodeInfo``.

The rename aliases the wire import as ``_WireNodeInfo`` inside
``cluster.py``. The module attribute ``NodeInfo`` is now absent,
so the shadow is closed.
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

    assert cluster_mod._WireNodeInfo is wire_responses.NodeInfo


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
