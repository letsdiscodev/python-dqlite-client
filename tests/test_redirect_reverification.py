"""``ClusterClient._verify_redirect`` re-probes a redirect target to
confirm it self-identifies as leader before trusting the hint.
Mirrors go-dqlite's ``connector.go::connectAttemptOne`` re-probe
(lines 285-294). A stale-state peer can hand back a node that no
longer holds leadership; without re-verification, the sweep returns
a stale hint and ``connect()`` wastes a full ``connect()``+Open round-
trip before retrying.
"""

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
    """Build a ``_query_leader`` mock that returns the configured
    leader for each address, raising the configured exception if
    listed. Records every call address."""
    raise_for = raise_for or {}
    calls: list[str] = []

    async def _impl(address: str, **_kw: object) -> str | None:
        calls.append(address)
        if address in raise_for:
            raise raise_for[address]
        return leader_for.get(address)

    _impl.calls = calls  # type: ignore[attr-defined]
    return _impl


# ---------------------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_redirect_verified_against_self() -> None:
    """Node A redirects to B; B confirms B as leader. The sweep must
    return B."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory(
        {
            "node-a:9001": "node-b:9001",  # A redirects to B
            "node-b:9001": "node-b:9001",  # B confirms itself
        }
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()
    assert leader == "node-b:9001"


# ---------------------------------------------------------------- stale hint


@pytest.mark.asyncio
async def test_redirect_stale_hint_falls_through() -> None:
    """Node A redirects to B; B reports C as leader (stale hint).
    The sweep must NOT trust B; with C also in the store and
    self-confirming, the result is C."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory(
        {
            "node-a:9001": "node-b:9001",  # A redirects to B
            "node-b:9001": "node-c:9001",  # B reports C (stale hint)
            "node-c:9001": "node-c:9001",  # C self-confirms
        }
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()
    assert leader == "node-c:9001"


@pytest.mark.asyncio
async def test_redirect_target_unreachable_falls_through() -> None:
    """A redirects to B; B is unreachable. The sweep must fall
    through; with no other valid leader, the call raises."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory(
        {"node-a:9001": "node-b:9001", "node-b:9001": None},
        raise_for={"node-b:9001": DqliteConnectionError("refused")},
    )
    cluster._query_leader = AsyncMock(side_effect=impl)

    with pytest.raises(ClusterError):
        await cluster.find_leader()


# ---------------------------------------------------------------- self-leader


@pytest.mark.asyncio
async def test_self_leader_does_not_trigger_verify() -> None:
    """When the queried node returns ITS OWN address as leader,
    that's not a redirect — verify must not run."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    impl = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl)

    leader = await cluster.find_leader()
    assert leader == "node-a:9001"
    # Exactly one call: the initial probe, no verify needed.
    assert impl.calls == ["node-a:9001"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------- verify on cache hit


@pytest.mark.asyncio
async def test_fast_path_cached_redirect_verified() -> None:
    """C3's fast-path probe of the cached leader: when the cached
    node responds with a redirect, the verify_redirect step must
    confirm the new target before trusting it."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    # Prime the cache with node-a.
    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()
    assert cluster._get_last_known_leader() == "node-a:9001"

    # Second call: cached node-a redirects to node-b; node-b
    # self-confirms. Cache must update to node-b.
    impl_second = _query_leader_factory(
        {"node-a:9001": "node-b:9001", "node-b:9001": "node-b:9001"}
    )
    cluster._query_leader = AsyncMock(side_effect=impl_second)
    leader = await cluster.find_leader()

    assert leader == "node-b:9001"
    assert cluster._get_last_known_leader() == "node-b:9001"


@pytest.mark.asyncio
async def test_fast_path_stale_redirect_clears_cache() -> None:
    """When the cached node redirects but verification fails (stale
    hint), the cache must be cleared and the full sweep must run."""
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    # Prime the cache with node-a.
    impl_first = _query_leader_factory({"node-a:9001": "node-a:9001"})
    cluster._query_leader = AsyncMock(side_effect=impl_first)
    await cluster.find_leader()

    # Second call: cached node-a redirects to node-b, but node-b
    # reports node-c (stale hint). Cache must clear and the full
    # sweep finds node-c.
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


# ---------------------------------------------------------------- _verify_redirect direct


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
    """The hinted node responds with no-leader-known. Treat as
    unverifiable and fall through."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(return_value=None)

    result = await cluster._verify_redirect("node-b:9001")
    assert result is None


@pytest.mark.asyncio
async def test_verify_redirect_returns_none_on_transport_error() -> None:
    """Verification failure is not fatal — the sweep falls through."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)
    cluster._query_leader = AsyncMock(side_effect=OSError("refused"))

    result = await cluster._verify_redirect("node-b:9001")
    assert result is None


@pytest.mark.asyncio
async def test_verify_redirect_logs_both_addresses_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On stale-hint, both responder and reported addresses appear
    in DEBUG logs (sanitised)."""
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
    """Pin: the method exists with the expected name."""
    assert hasattr(ClusterClient, "_verify_redirect")
