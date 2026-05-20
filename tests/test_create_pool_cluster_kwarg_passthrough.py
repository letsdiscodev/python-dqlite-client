"""Pin: ``ConnectionPool`` / ``create_pool`` forwards
``concurrent_leader_conns`` and ``redirect_policy`` into the
auto-built ``ClusterClient``, AND raises ``ValueError`` when these
kwargs are combined with an externally-owned ``cluster=``.

Mirrors the existing ``dial_func`` mutual-exclusion + passthrough
pattern. A regression dropping either ``cluster_kwargs[...] = ...``
injection silently strips the operator-supplied knob; a regression
softening the mutual-exclusion would silently desync the pool's
claimed configuration from the externally-owned cluster's actual
behaviour.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


def test_pool_passes_concurrent_leader_conns_through_to_auto_built_cluster() -> None:
    """Without ``cluster=``, the pool builds an internal
    ``ClusterClient``; the kwarg must be forwarded into the
    constructor so the operator's tuning is preserved."""
    pool = ConnectionPool(["a:9001"], concurrent_leader_conns=4)
    assert pool._cluster._concurrent_leader_conns == 4


def test_pool_passes_redirect_policy_through_to_auto_built_cluster() -> None:
    """Same passthrough discipline for the redirect policy callable."""

    def policy(addr: str) -> bool:
        return True

    pool = ConnectionPool(["a:9001"], redirect_policy=policy)
    assert pool._cluster._redirect_policy is policy


def test_pool_rejects_concurrent_leader_conns_with_external_cluster() -> None:
    """An externally-owned ``cluster=`` carries its own
    configuration. Allowing the pool kwarg here would silently
    desync the pool's claimed configuration from the cluster's
    actual behaviour."""
    cluster = ClusterClient(MemoryNodeStore(addresses=["a:9001"]))
    with pytest.raises(ValueError, match="concurrent_leader_conns cannot be combined"):
        ConnectionPool(["a:9001"], cluster=cluster, concurrent_leader_conns=4)


def test_pool_rejects_redirect_policy_with_external_cluster() -> None:
    """Same mutual-exclusion discipline for the redirect policy."""
    cluster = ClusterClient(MemoryNodeStore(addresses=["a:9001"]))

    def policy(addr: str) -> bool:
        return True

    with pytest.raises(ValueError, match="redirect_policy cannot be combined"):
        ConnectionPool(["a:9001"], cluster=cluster, redirect_policy=policy)
