"""Pin: two symmetric paths that previously left a stale leader in
``_last_known_leader`` now both invalidate the cache:

1. ``ClusterClient.try_connect``'s per-attempt failure arm: when
   ``find_leader`` returned a leader but the subsequent connect /
   handshake / ``open_database`` failed, the next retry's
   ``find_leader`` would fast-path probe the same now-stale
   leader. Mirrors go-dqlite's ``connector.go::Connect`` clearing
   the leader tracker on per-attempt failure.

2. ``ConnectionPool.acquire``'s dead-conn drain arm: when an idle
   conn dequeued from the pool trips ``_socket_looks_dead``, the
   pool runs ``_drain_idle`` AND then invalidates the cluster's
   leader cache so the immediate ``_create_connection`` below
   doesn't pay one stale-leader probe RTT.

Both arms are gated on the leader-was-known signal — gratuitous
invalidation when ``find_leader`` itself failed is avoided (it
already cleared the cache internally).
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


def test_try_connect_failure_arm_invalidates_leader_cache_source() -> None:
    """Inspection pin: the per-attempt failure arm clears the cache
    when ``leader is not None``. A regression dropping the
    invalidation would let the next retry attempt fast-path probe
    the stale leader."""
    src = inspect.getsource(ClusterClient.connect)
    assert "_set_last_known_leader(None)" in src
    # Must be gated so a failed find_leader doesn't redundantly
    # re-clear an already-cleared cache.
    assert "if leader is not None:" in src


def test_pool_dead_conn_drain_arm_invalidates_leader_cache_source() -> None:
    """Inspection pin on the pool: after ``_drain_idle`` runs (the
    sibling-evidence path), the cluster's leader cache is cleared
    before ``_create_connection`` so the create's first
    ``find_leader`` runs a fresh sweep."""
    src = inspect.getsource(ConnectionPool.acquire)
    # Both invalidations live in the same module — but the pool
    # invalidation reaches into the cluster's slot.
    assert "_set_last_known_leader(None)" in src
