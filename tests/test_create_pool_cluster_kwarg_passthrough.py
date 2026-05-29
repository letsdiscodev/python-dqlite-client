"""ConnectionPool forwards concurrent_leader_conns/redirect_policy into the
auto-built ClusterClient, and rejects them when combined with an external cluster=."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


def test_pool_passes_concurrent_leader_conns_through_to_auto_built_cluster() -> None:
    """Without cluster=, the kwarg is forwarded into the auto-built ClusterClient."""
    pool = ConnectionPool(["a:9001"], concurrent_leader_conns=4)
    assert pool._cluster._concurrent_leader_conns == 4


def test_pool_passes_redirect_policy_through_to_auto_built_cluster() -> None:
    """Same passthrough for the redirect policy."""

    def policy(addr: str) -> bool:
        return True

    pool = ConnectionPool(["a:9001"], redirect_policy=policy)
    assert pool._cluster._redirect_policy is policy


def test_pool_rejects_concurrent_leader_conns_with_external_cluster() -> None:
    """An external cluster= carries its own config; the pool kwarg would desync it."""
    cluster = ClusterClient(MemoryNodeStore(addresses=["a:9001"]))
    with pytest.raises(ValueError, match="concurrent_leader_conns cannot be combined"):
        ConnectionPool(["a:9001"], cluster=cluster, concurrent_leader_conns=4)


def test_pool_rejects_redirect_policy_with_external_cluster() -> None:
    """Same mutual exclusion for the redirect policy."""
    cluster = ClusterClient(MemoryNodeStore(addresses=["a:9001"]))

    def policy(addr: str) -> bool:
        return True

    with pytest.raises(ValueError, match="redirect_policy cannot be combined"):
        ConnectionPool(["a:9001"], cluster=cluster, redirect_policy=policy)
