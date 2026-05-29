"""Two paths invalidate ``_last_known_leader`` to avoid a stale-leader probe:
the connect per-attempt failure arm and the pool's dead-conn drain arm. Both are
gated on leader-was-known so a failed ``find_leader`` (which already cleared the
cache) is not re-cleared."""

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
    """The per-attempt failure arm clears the cache when ``leader is not None``."""
    src = inspect.getsource(ClusterClient.connect)
    assert "_set_last_known_leader(None)" in src
    assert "if leader is not None:" in src


def test_pool_dead_conn_drain_arm_invalidates_leader_cache_source() -> None:
    """After ``_drain_idle`` runs, the cluster's leader cache is cleared before
    ``_create_connection`` so the next ``find_leader`` sweeps fresh."""
    src = inspect.getsource(ConnectionPool.acquire)
    assert "_set_last_known_leader(None)" in src


@pytest.mark.asyncio
async def test_connect_retry_failure_arm_clears_stale_leader_cache() -> None:
    """Between attempts the cache must be cleared: the second ``find_leader``
    observes ``None`` at entry rather than the seeded stale leader."""
    store = MemoryNodeStore(["seed:9001"])
    cluster = ClusterClient(store, timeout=2.0)
    cluster._set_last_known_leader("stale:9001")

    leader_calls_at_entry: list[str | None] = []

    async def _record_and_return(**_kw: Any) -> str:
        leader_calls_at_entry.append(cluster._get_last_known_leader())
        return "stale:9001"

    def _fake_dqlite_connection_ctor(*_args: Any, **_kwargs: Any) -> Any:
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
        await asyncio.wait_for(cluster.connect(max_attempts=2), timeout=2.0)

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
    """When ``find_leader`` itself raises, the ``leader is not None`` gate skips
    the failure arm's ``_set_last_known_leader(None)`` (pinned via call count)."""
    store = MemoryNodeStore(["seed:9001"])
    cluster = ClusterClient(store, timeout=2.0)
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

    assert all(call is None or call is not None for call in set_calls), (
        f"Set calls observed: {set_calls!r}"
    )
    # find_leader is patched to raise before its body, so any None call here
    # comes exclusively from the per-attempt failure arm under test.
    none_calls = [c for c in set_calls if c is None]
    assert none_calls == [], (
        f"Per-attempt failure arm must skip _set_last_known_leader(None) "
        f"when find_leader itself failed (leader is None); got "
        f"{len(none_calls)} redundant invalidations: {set_calls!r}"
    )


@pytest.mark.asyncio
async def test_pool_dead_conn_drain_arm_clears_cluster_leader_cache() -> None:
    """The pool dead-conn drain arm clears the cluster's leader cache before
    ``_create_connection`` runs."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=2, timeout=1.0)

    pool._cluster._set_last_known_leader("stale:9001")
    assert pool._cluster._get_last_known_leader() == "stale:9001"

    dead = MagicMock()
    dead.is_connected = False
    dead._pool_released = True
    dead._protocol = None
    dead.close = AsyncMock(return_value=None)
    pool._pool.put_nowait(dead)
    pool._size = 1  # reservation slot for the queued conn

    fresh = MagicMock()
    fresh.is_connected = True
    fresh._pool_released = False
    fresh._protocol = MagicMock()
    fresh.close = AsyncMock(return_value=None)

    async def _make_fresh() -> Any:
        return fresh

    pool._create_connection = AsyncMock(side_effect=_make_fresh)

    async with pool.acquire() as conn:
        assert conn is fresh

    assert pool._cluster._get_last_known_leader() is None, (
        "Pool dead-conn drain arm must invalidate the cluster's "
        "leader cache so the next sweep runs fresh"
    )


@pytest.mark.asyncio
async def test_pool_dead_conn_drain_arm_tolerates_missing_cluster_attr() -> None:
    """The ``getattr(self, "_cluster", None)`` tolerance keeps the dead-conn arm
    from raising ``AttributeError`` when ``_cluster`` is absent."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=2, timeout=1.0)
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

    async with pool.acquire() as conn:
        assert conn is fresh
