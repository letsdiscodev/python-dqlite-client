"""``ConnectionPool(node_store=...)`` forwards ``redirect_policy`` (SSRF protection)
and ``concurrent_leader_conns`` to the auto-built ``ClusterClient`` — a separate
branch from the ``addresses=`` path."""

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
