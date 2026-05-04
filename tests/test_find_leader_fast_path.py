"""``ClusterClient._find_leader_impl`` consults a single-entry
last-known-leader cache before running the parallel sweep. Mirrors
go-dqlite's ``LeaderTracker`` fast path (``connector.go:228-237``):
on hit, one probe replaces the full sweep; on miss, the cache is
cleared and the sweep runs as normal.

The cache is purely internal — no public kwargs grow.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DqliteConnectionError,
)


def _query_leader_factory(
    leader_for: dict[str, str | None],
    *,
    raise_for: dict[str, Exception] | None = None,
) -> Callable[..., object]:
    """Build a ``_query_leader`` mock that returns / raises per
    address. Records every call for assertion."""
    raise_for = raise_for or {}
    calls: list[str] = []

    async def _impl(address: str, **_kw: object) -> str | None:
        calls.append(address)
        if address in raise_for:
            raise raise_for[address]
        return leader_for.get(address)

    _impl.calls = calls  # type: ignore[attr-defined]
    return _impl


@pytest.mark.asyncio
async def test_initial_state_cache_is_none() -> None:
    cluster = ClusterClient.from_addresses(["localhost:9001"])
    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_first_call_runs_full_sweep_then_caches_leader() -> None:
    """On a fresh cluster client the cache is None, so the first
    ``find_leader`` runs the full sweep. The success path must
    populate the cache with the discovered leader address."""
    addresses = ["node-a:9001", "node-b:9001", "node-c:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    leader_responses: dict[str, str | None] = {
        "node-a:9001": None,
        "node-b:9001": None,
        "node-c:9001": "node-c:9001",
    }
    impl = _query_leader_factory(leader_responses)
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()

    assert leader == "node-c:9001"
    assert cluster._get_last_known_leader() == "node-c:9001"


@pytest.mark.asyncio
async def test_second_call_takes_fast_path_one_probe_only() -> None:
    """After a successful sweep, the second ``find_leader`` should
    issue exactly one probe (against the cached address)."""
    addresses = ["node-a:9001", "node-b:9001", "node-c:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    leader_responses: dict[str, str | None] = {
        "node-a:9001": None,
        "node-b:9001": "node-b:9001",
        "node-c:9001": None,
    }
    impl = _query_leader_factory(leader_responses)
    cluster._query_leader = AsyncMock(side_effect=impl)

    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-b:9001"

    # Reset the call log; second call should hit only the cache.
    impl.calls.clear()  # type: ignore[attr-defined]
    leader = await cluster.find_leader()

    assert leader == "node-b:9001"
    assert impl.calls == ["node-b:9001"], (  # type: ignore[attr-defined]
        f"expected exactly one fast-path probe, got {impl.calls}"  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_fast_path_failure_clears_cache_and_falls_through() -> None:
    """If the cached leader's probe raises a transport error, the
    cache must be cleared and the full sweep must run, populating
    the cache with the new leader's address."""
    addresses = ["node-a:9001", "node-b:9001", "node-c:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    # First call: leader is node-b.
    impl_first = _query_leader_factory(
        {"node-a:9001": None, "node-b:9001": "node-b:9001", "node-c:9001": None}
    )
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-b:9001"

    # Second call: cached node-b is now down, leader has moved to
    # node-a. The fast-path probe must fail, the cache must clear,
    # and the sweep must rediscover the new leader.
    impl_second = _query_leader_factory(
        {"node-a:9001": "node-a:9001", "node-b:9001": None, "node-c:9001": None},
        raise_for={"node-b:9001": DqliteConnectionError("connection refused")},
    )
    cluster._query_leader = AsyncMock(side_effect=impl_second)

    leader = await cluster.find_leader()

    assert leader == "node-a:9001"
    assert cluster._get_last_known_leader() == "node-a:9001"
    # Sequence: cached fast-path probe of node-b first, then full sweep.
    assert impl_second.calls[0] == "node-b:9001"  # type: ignore[attr-defined]
    assert "node-a:9001" in impl_second.calls  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fast_path_no_leader_known_falls_through() -> None:
    """If the cached node responds with no-leader-known, the cache
    must be cleared and the full sweep must run."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001", "node-b:9001": None})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-a:9001"

    # Second call: node-a now reports no-leader-known; node-b is
    # the new leader.
    impl_second = _query_leader_factory({"node-a:9001": None, "node-b:9001": "node-b:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_second)

    leader = await cluster.find_leader()

    assert leader == "node-b:9001"
    assert cluster._get_last_known_leader() == "node-b:9001"


@pytest.mark.asyncio
async def test_fast_path_redirect_updates_cache_to_redirect_target() -> None:
    """When the cached node redirects to a different leader, the
    cache must be updated to the redirect target (the actual
    leader), not the responder."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    # Prime the cache with node-a.
    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001", "node-b:9001": None})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-a:9001"

    # Second call: node-a is no longer leader and redirects to node-b.
    impl_second = _query_leader_factory(
        {"node-a:9001": "node-b:9001", "node-b:9001": "node-b:9001"}
    )
    cluster._query_leader = AsyncMock(side_effect=impl_second)

    leader = await cluster.find_leader()

    assert leader == "node-b:9001"
    # Cache must be the actual leader, not the redirector.
    assert cluster._get_last_known_leader() == "node-b:9001"


@pytest.mark.asyncio
async def test_fast_path_redirect_policy_clears_cache_and_propagates() -> None:
    """A redirect to a forbidden address from the cached node must
    clear the cache AND propagate ``ClusterPolicyError``. Falling
    through would not help — the same policy applies elsewhere."""
    addresses = ["node-a:9001", "node-b:9001"]

    def deny_all(_address: str) -> bool:
        return False

    cluster = ClusterClient.from_addresses(addresses, timeout=5.0, redirect_policy=deny_all)

    # Prime cache: with deny-all policy active, prime via direct
    # attribute write to skip the policy on the first call.
    cluster._set_last_known_leader("node-a:9001")

    impl = _query_leader_factory({"node-a:9001": "redirect-target:9001", "node-b:9001": None})
    cluster._query_leader = AsyncMock(side_effect=impl)

    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()

    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_fast_path_probe() -> None:
    """The single-flight slot map collapses concurrent
    ``find_leader`` callers across calls — the fast-path probe lives
    inside ``_find_leader_impl``, so 10 simultaneous callers share
    one probe (caller A's), not ten."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    # Prime the cache.
    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-a:9001"

    # 10 concurrent callers — all should collapse onto one probe.
    impl_second = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_second)

    callers = [asyncio.create_task(cluster.find_leader()) for _ in range(10)]
    results = await asyncio.gather(*callers)

    assert all(r == "node-a:9001" for r in results)
    assert len(impl_second.calls) == 1, (  # type: ignore[attr-defined]
        f"expected one shared fast-path probe across 10 callers; got {len(impl_second.calls)}"  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_cache_update_atomic_on_success() -> None:
    """Cache reflects most recent successful return — a sequence of
    ``find_leader`` calls against different leaders must end with
    the cache pointing at the latest one."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    # First leader: node-a.
    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-a:9001"

    # Leader flips: node-b. node-a now reports no-leader-known.
    impl_second = _query_leader_factory({"node-a:9001": None, "node-b:9001": "node-b:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_second)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-b:9001"


@pytest.mark.asyncio
async def test_fast_path_with_timeout_clears_cache() -> None:
    """A TimeoutError on the fast-path probe must clear the cache
    and trigger fall-through."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=0.5)

    # Prime cache with node-a.
    cluster._set_last_known_leader("node-a:9001")

    impl = _query_leader_factory(
        {"node-a:9001": None, "node-b:9001": "node-b:9001"},
        raise_for={"node-a:9001": TimeoutError("probe timed out")},
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()

    assert leader == "node-b:9001"
    assert cluster._get_last_known_leader() == "node-b:9001"


@pytest.mark.asyncio
async def test_fast_path_skipped_when_cache_is_none() -> None:
    """Sanity check: with a fresh cache, the fast path must NOT
    issue a probe. The full sweep is the first thing to run."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    impl = _query_leader_factory({"node-a:9001": "node-a:9001", "node-b:9001": None})
    cluster._query_leader = AsyncMock(side_effect=impl)

    await cluster.find_leader()

    # Both probes ran in parallel — no fast-path probe preceded them.
    # We can't easily distinguish "no fast path" from "fast path then
    # full sweep", but we CAN assert that node-a appears exactly once
    # (not once-fast-path + once-in-sweep).
    assert impl.calls.count("node-a:9001") == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_aggregate_failure_does_not_populate_cache() -> None:
    """If every probe fails, the cache must remain ``None`` — no
    successful winning_address, no cache update."""
    addresses = ["node-a:9001", "node-b:9001"]
    cluster = ClusterClient.from_addresses(addresses, timeout=5.0)

    impl = _query_leader_factory(
        {},
        raise_for={
            "node-a:9001": DqliteConnectionError("refused on a"),
            "node-b:9001": DqliteConnectionError("refused on b"),
        },
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    with pytest.raises(ClusterError):
        await cluster.find_leader()

    assert cluster._get_last_known_leader() is None
