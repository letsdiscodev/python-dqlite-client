"""Pin: ``ConnectionPool(node_store=...)`` forwards ``redirect_policy``
and ``concurrent_leader_conns`` to the auto-built ``ClusterClient``.

This is a separate branch from the ``addresses=`` path (the latter is
covered by ``test_create_pool_cluster_kwarg_passthrough.py``). A
regression that dropped the injection in the ``node_store`` branch
would silently strip the operator's redirect (SSRF-protection) policy
and connection-budget tuning while the addresses-path tests still pass.
"""

from __future__ import annotations

from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


def test_pool_node_store_branch_forwards_redirect_policy_and_clc() -> None:
    def policy(addr: str) -> bool:
        return True

    store = MemoryNodeStore(addresses=["10.0.0.1:9001"])
    pool = ConnectionPool(
        node_store=store,
        redirect_policy=policy,
        concurrent_leader_conns=4,
    )

    assert pool._cluster._redirect_policy is policy
    assert pool._cluster._concurrent_leader_conns == 4
