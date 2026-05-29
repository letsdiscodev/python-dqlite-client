"""``ClusterClient._verify_redirect`` re-probes a redirect target to confirm it
self-identifies as leader before trusting a possibly-stale hint
(mirrors go-dqlite's ``connector.go::connectAttemptOne`` re-probe)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterError,
    DqliteConnectionError,
)
from dqliteclient.node_store import MemoryNodeStore


def _query_leader_factory(
    leader_for: dict[str, str | None],
    *,
    raise_for: dict[str, Exception] | None = None,
) -> Callable[..., object]:
    """Build a ``_query_leader`` mock that returns/raises per address and records calls."""
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
async def test_redirect_verified_against_self() -> None:
    """A redirects to B; B confirms itself, so the sweep returns B."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory(
        {
            "node-a:9001": "node-b:9001",
            "node-b:9001": "node-b:9001",
        }
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()
    assert leader == "node-b:9001"


@pytest.mark.asyncio
async def test_redirect_stale_hint_falls_through() -> None:
    """A redirects to B; B reports C (stale hint), so the sweep falls through to C."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory(
        {
            "node-a:9001": "node-b:9001",
            "node-b:9001": "node-c:9001",  # stale hint
            "node-c:9001": "node-c:9001",
        }
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()
    assert leader == "node-c:9001"


@pytest.mark.asyncio
async def test_redirect_target_unreachable_falls_through() -> None:
    """A redirects to B; B is unreachable and no other leader exists, so the call raises."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory(
        {"node-a:9001": "node-b:9001", "node-b:9001": None},
        raise_for={"node-b:9001": DqliteConnectionError("refused")},
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    with pytest.raises(ClusterError):
        await cluster.find_leader()


@pytest.mark.asyncio
async def test_self_leader_does_not_trigger_verify() -> None:
    """A node returning its own address is not a redirect, so verify must not run."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()
    assert leader == "node-a:9001"
    assert impl.calls == ["node-a:9001"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fast_path_cached_redirect_verified() -> None:
    """Fast-path: a cached node that redirects must have its new target verified."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-a:9001"

    # Cached node-a redirects to node-b, which self-confirms; cache must update to node-b.
    impl_second = _query_leader_factory(
        {"node-a:9001": "node-b:9001", "node-b:9001": "node-b:9001"}
    )
    cluster._query_leader = AsyncMock(side_effect=impl_second)
    leader = await cluster.find_leader()

    assert leader == "node-b:9001"
    assert cluster._get_last_known_leader() == "node-b:9001"


@pytest.mark.asyncio
async def test_fast_path_stale_redirect_clears_cache() -> None:
    """When the cached node's redirect fails verification, the cache clears and the
    full sweep runs."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()

    # Cached node-a redirects to node-b, but node-b reports node-c (stale hint).
    impl_second = _query_leader_factory(
        {
            "node-a:9001": "node-b:9001",
            "node-b:9001": "node-c:9001",
            "node-c:9001": "node-c:9001",
        }
    )
    cluster._query_leader = AsyncMock(side_effect=impl_second)
    leader = await cluster.find_leader()

    assert leader == "node-c:9001"
    assert cluster._get_last_known_leader() == "node-c:9001"


@pytest.mark.asyncio
async def test_verify_redirect_returns_address_on_self_confirmation() -> None:
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(return_value="node-b:9001")

    result = await cluster._verify_redirect("node-b:9001")
    assert result == "node-b:9001"


@pytest.mark.asyncio
async def test_verify_redirect_returns_none_on_address_mismatch() -> None:
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(return_value="node-c:9001")

    result = await cluster._verify_redirect("node-b:9001")
    assert result is None


@pytest.mark.asyncio
async def test_verify_redirect_returns_none_on_no_leader_known() -> None:
    """A no-leader-known response is unverifiable; fall through."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(return_value=None)

    result = await cluster._verify_redirect("node-b:9001")
    assert result is None


@pytest.mark.asyncio
async def test_verify_redirect_returns_none_on_transport_error() -> None:
    """A transport error during verification is not fatal; fall through."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(side_effect=OSError("refused"))

    result = await cluster._verify_redirect("node-b:9001")
    assert result is None


@pytest.mark.asyncio
async def test_verify_redirect_logs_both_addresses_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On stale-hint, both responder and reported addresses appear in DEBUG logs."""
    import logging as _logging

    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(return_value="node-c:9001")

    with caplog.at_level(_logging.DEBUG, logger="dqliteclient.cluster"):
        await cluster._verify_redirect("node-b:9001")

    msgs = [r.getMessage() for r in caplog.records]
    assert any("node-b:9001" in m and "node-c:9001" in m and "stale hint" in m for m in msgs), (
        f"expected stale-hint log with both addresses, got: {msgs}"
    )


def test_verify_redirect_method_exists() -> None:
    assert hasattr(ClusterClient, "_verify_redirect")
