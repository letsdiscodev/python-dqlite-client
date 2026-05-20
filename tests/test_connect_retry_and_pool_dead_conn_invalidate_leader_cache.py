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

import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore
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


@pytest.mark.asyncio
async def test_connect_retry_failure_arm_clears_stale_leader_cache() -> None:
    """Behavioural pin for the connect-retry leader-cache invalidation.

    Seed ``_last_known_leader = "stale:9001"``. Patch ``find_leader``
    to record the cache state at entry, then return "stale:9001".
    Patch the ``DqliteConnection`` constructor to return a mock whose
    ``connect()`` raises ``DqliteConnectionError`` (the canonical
    transport-failure path the retry loop catches).

    Pin: between attempts, the cache must be cleared. The second
    ``find_leader`` call observes ``None`` at entry; without the
    invalidation, it would see the stale "stale:9001" and waste one
    fast-path RTT against the dead leader per retry. Mirrors
    go-dqlite's ``connector.go::Connect`` discipline.
    """
    store = MemoryNodeStore(["seed:9001"])
    cluster = ClusterClient(store, timeout=2.0)
    cluster._set_last_known_leader("stale:9001")

    leader_calls_at_entry: list[str | None] = []

    async def _record_and_return(**_kw: Any) -> str:
        # Snapshot the cache BEFORE the find_leader body runs so the
        # cleared / not-cleared state is observable per attempt.
        leader_calls_at_entry.append(cluster._get_last_known_leader())
        return "stale:9001"

    def _fake_dqlite_connection_ctor(*_args: Any, **_kwargs: Any) -> Any:
        # Return a fresh mock each call so the retry loop sees a
        # distinct conn per attempt.
        m = MagicMock()
        m.connect = AsyncMock(side_effect=DqliteConnectionError("synthetic transport failure"))
        m.close = AsyncMock(return_value=None)
        return m

    import dqliteclient.cluster as _cluster_mod

    with (
        patch.object(cluster, "find_leader", side_effect=_record_and_return),
        patch.object(_cluster_mod, "DqliteConnection", _fake_dqlite_connection_ctor),
        pytest.raises(DqliteConnectionError),
    ):
        # max_attempts=2 keeps the test cheap; the assertion is about
        # the cache state observed at the SECOND attempt's entry.
        await asyncio.wait_for(cluster.connect(max_attempts=2), timeout=2.0)

    # First attempt entered with the seeded stale cache; second
    # attempt MUST see the cleared cache. Regression that drops
    # _set_last_known_leader(None) in the failure arm would show
    # ["stale:9001", "stale:9001"] here.
    assert len(leader_calls_at_entry) >= 2, (
        f"Expected at least two retry attempts, got {len(leader_calls_at_entry)}"
    )
    assert leader_calls_at_entry[0] == "stale:9001", (
        f"First attempt should see the seeded cache: {leader_calls_at_entry!r}"
    )
    assert leader_calls_at_entry[1] is None, (
        f"Second attempt MUST see a cleared cache (the per-attempt "
        f"failure arm must call _set_last_known_leader(None)): "
        f"{leader_calls_at_entry!r}"
    )


@pytest.mark.asyncio
async def test_connect_retry_skips_invalidate_when_find_leader_failed() -> None:
    """Gated arm pin: when ``find_leader`` itself raises, the
    ``leader is not None`` gate keeps the failure arm from
    redundantly re-clearing an already-cleared cache.

    Seed the cache as ``None`` (the post-find_leader-failure state).
    Drive ``find_leader`` to raise ``DqliteConnectionError`` on every
    attempt. The failure arm catches the exception, but the gate
    must skip the ``_set_last_known_leader(None)`` call because
    ``leader is None`` (assignment never happened).

    Behavioural assertion: the cache stays None throughout. The
    interesting regression is the inverse — a regression dropping
    the gate would still pass since the call is a no-op on None,
    but the gate is the documented "don't redundantly re-clear"
    discipline. We pin it by checking the call count on a wrapped
    ``_set_last_known_leader`` instead.
    """
    store = MemoryNodeStore(["seed:9001"])
    cluster = ClusterClient(store, timeout=2.0)
    # Pre-clear so the cache observed at the failure arm is None.
    cluster._set_last_known_leader(None)

    set_calls: list[str | None] = []
    original_setter = cluster._set_last_known_leader

    def _track_set(addr: str | None) -> None:
        set_calls.append(addr)
        original_setter(addr)

    async def _always_raise(**_kw: Any) -> str:
        raise DqliteConnectionError("synthetic find_leader failure")

    with (
        patch.object(cluster, "_set_last_known_leader", side_effect=_track_set),
        patch.object(cluster, "find_leader", side_effect=_always_raise),
        pytest.raises(DqliteConnectionError),
    ):
        await asyncio.wait_for(cluster.connect(max_attempts=2), timeout=2.0)

    # The failure arm's gate ``if leader is not None: ...`` must
    # keep _set_last_known_leader(None) from firing on the
    # find_leader-itself-failed path. Pre-fix path would call it
    # unconditionally; the gate makes that call count == 0 here.
    assert all(call is None or call is not None for call in set_calls), (
        f"Set calls observed: {set_calls!r}"
    )
    # Stronger: no None invalidations originated from the failure
    # arm — find_leader's own internal cache-clears would only fire
    # if the body ran, but we patched find_leader to raise BEFORE
    # the body executes. Any None call here would come exclusively
    # from the per-attempt failure arm we're pinning.
    none_calls = [c for c in set_calls if c is None]
    assert none_calls == [], (
        f"Per-attempt failure arm must skip _set_last_known_leader(None) "
        f"when find_leader itself failed (leader is None); got "
        f"{len(none_calls)} redundant invalidations: {set_calls!r}"
    )


@pytest.mark.asyncio
async def test_pool_dead_conn_drain_arm_clears_cluster_leader_cache() -> None:
    """Behavioural pin for the pool dead-conn drain arm's reach into
    the cluster's leader cache.

    Stage an idle conn whose ``is_connected == False`` (the simpler
    of the two dead-conn predicates — short-circuits before
    ``_socket_looks_dead``). Seed
    ``cluster._set_last_known_leader("stale:9001")``. Drive
    ``pool.acquire()`` with ``_create_connection`` patched to return
    a fresh mock. Assert: after the dead-conn arm runs,
    ``cluster._get_last_known_leader() is None``.

    Pre-fix: a leader-flip would land every pool acquirer through
    one wasted stale-leader RTT before the fresh sweep. Without
    this pin a regression dropping the ``cluster._set_last_known_leader(None)``
    line would re-introduce the linear-in-queue-depth slowdown.
    """
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=2, timeout=1.0)

    # Seed the cluster's cache as if a prior call had cached the
    # now-dead leader.
    pool._cluster._set_last_known_leader("stale:9001")
    assert pool._cluster._get_last_known_leader() == "stale:9001"

    # Stage a dead idle conn.
    dead = MagicMock()
    dead.is_connected = False
    dead._pool_released = True
    dead._protocol = None
    dead.close = AsyncMock(return_value=None)
    pool._pool.put_nowait(dead)
    pool._size = 1  # reservation slot for the queued conn

    # Patch _create_connection so the test does not actually dial.
    fresh = MagicMock()
    fresh.is_connected = True
    fresh._pool_released = False
    fresh._protocol = MagicMock()
    fresh.close = AsyncMock(return_value=None)

    async def _make_fresh() -> Any:
        return fresh

    pool._create_connection = AsyncMock(side_effect=_make_fresh)

    async with pool.acquire() as conn:
        # The yielded conn is the freshly created one.
        assert conn is fresh

    # CRITICAL pin: the cluster's leader cache must be cleared by
    # the dead-conn drain arm before _create_connection runs. A
    # regression that drops the reach (or replaces it with a no-op)
    # would leave "stale:9001" in place here.
    assert pool._cluster._get_last_known_leader() is None, (
        "Pool dead-conn drain arm must invalidate the cluster's "
        "leader cache so the next sweep runs fresh"
    )


@pytest.mark.asyncio
async def test_pool_dead_conn_drain_arm_tolerates_missing_cluster_attr() -> None:
    """Defence-in-depth pin: the ``getattr(self, "_cluster", None)``
    tolerance keeps the dead-conn arm from crashing on
    ``AttributeError`` for fixtures that drop the ``_cluster``
    attribute.

    Build a regular pool, then ``del pool._cluster`` to simulate a
    bypass. Drive the dead-conn arm; ``getattr`` returns ``None``,
    the ``if cluster is not None:`` gate skips the reach, and the
    arm completes without raising. A regression replacing
    ``getattr(self, "_cluster", None)`` with ``self._cluster``
    would raise ``AttributeError`` here.
    """
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=2, timeout=1.0)
    # Drop _cluster to simulate the __new__-built fixture pattern.
    del pool._cluster

    dead = MagicMock()
    dead.is_connected = False
    dead._pool_released = True
    dead._protocol = None
    dead.close = AsyncMock(return_value=None)
    pool._pool.put_nowait(dead)
    pool._size = 1

    fresh = MagicMock()
    fresh.is_connected = True
    fresh._pool_released = False
    fresh._protocol = MagicMock()
    fresh.close = AsyncMock(return_value=None)

    async def _make_fresh() -> Any:
        return fresh

    pool._create_connection = AsyncMock(side_effect=_make_fresh)

    # If the dead-conn arm reached self._cluster directly (without
    # the getattr tolerance), this would raise AttributeError. The
    # getattr-with-None-default + ``if cluster is not None:`` gate
    # keeps the arm operational.
    async with pool.acquire() as conn:
        assert conn is fresh
