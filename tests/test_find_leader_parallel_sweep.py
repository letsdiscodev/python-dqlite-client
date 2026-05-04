"""``ClusterClient._find_leader_impl`` probes nodes in parallel,
bounded by ``concurrent_leader_conns``. Mirrors go-dqlite's
``connector.go::connectAttemptAll`` first-success-wins semantic. The
serial pre-image probed nodes one at a time, so a leader sitting at
the back of a multi-node store paid O(N × timeout) wall clock under a
flip; the parallel sweep bounds it at one probe time.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DqliteConnectionError,
)
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_parallel_sweep_returns_first_success_under_slow_peers() -> None:
    """5 nodes, four sleep 5 s, one returns immediately. Wall clock
    must be bounded by the fastest probe, not the slowest."""
    addresses = [f"slow-{i}:9001" for i in range(4)] + ["fast:9001"]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0)

    async def _query_leader(address: str, **_kw: object) -> str:
        if address == "fast:9001":
            return "fast:9001"
        await asyncio.sleep(5.0)
        return ""

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    start = time.monotonic()
    leader = await cluster.find_leader()
    elapsed = time.monotonic() - start

    assert leader == "fast:9001"
    assert elapsed < 1.0, f"parallel sweep took {elapsed:.3f}s; expected < 1s"


@pytest.mark.asyncio
async def test_parallel_sweep_caps_concurrency_at_default_10() -> None:
    """50 nodes, count concurrent in-flight probes via a barrier; the
    cap must hold at the default of 10."""
    addresses = [f"node-{i}:9001" for i in range(50)]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0)

    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def _query_leader(address: str, **_kw: object) -> str | None:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await release.wait()
            return None  # no-leader-known so the sweep continues
        finally:
            in_flight -= 1

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    sweep = asyncio.create_task(cluster.find_leader())
    # Yield enough times for all 10 first-batch probes to enter and
    # the rest to queue on the semaphore.
    for _ in range(20):
        await asyncio.sleep(0)

    captured = max_in_flight
    release.set()
    with pytest.raises(ClusterError):
        await sweep

    assert captured == 10, f"expected max_in_flight == 10 at default cap; got {captured}"


@pytest.mark.asyncio
async def test_parallel_sweep_caps_concurrency_respects_kwarg() -> None:
    """When ``concurrent_leader_conns=3`` is passed, the cap must
    follow."""
    addresses = [f"node-{i}:9001" for i in range(20)]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0, concurrent_leader_conns=3)

    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def _query_leader(address: str, **_kw: object) -> str | None:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await release.wait()
            return None
        finally:
            in_flight -= 1

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    sweep = asyncio.create_task(cluster.find_leader())
    for _ in range(20):
        await asyncio.sleep(0)

    captured = max_in_flight
    release.set()
    with pytest.raises(ClusterError):
        await sweep

    assert captured == 3, f"expected max_in_flight == 3; got {captured}"


@pytest.mark.parametrize(
    "value",
    [
        True,  # bool slips through isinstance(_, int) without a guard
        False,
        0,
        -1,
    ],
)
def test_concurrent_leader_conns_validation(value: object) -> None:
    """Reject ``bool`` and ``< 1`` so the cap can never silently
    coerce to ``0`` or be misinterpreted."""
    store = MemoryNodeStore(["localhost:9001"])
    with pytest.raises((TypeError, ValueError)):
        ClusterClient(store, concurrent_leader_conns=value)  # type: ignore[arg-type]


def test_concurrent_leader_conns_bool_rejected_with_typeerror() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    with pytest.raises(TypeError, match="concurrent_leader_conns"):
        ClusterClient(store, concurrent_leader_conns=True)


def test_concurrent_leader_conns_passthrough_via_from_addresses() -> None:
    """``from_addresses`` must forward ``concurrent_leader_conns`` to
    the underlying ``ClusterClient`` so callers using the convenience
    constructor get the same knob."""
    cluster = ClusterClient.from_addresses(["localhost:9001"], concurrent_leader_conns=4)
    assert cluster._concurrent_leader_conns == 4


def test_concurrent_leader_conns_default_is_10() -> None:
    cluster = ClusterClient.from_addresses(["localhost:9001"])
    assert cluster._concurrent_leader_conns == 10


def test_concurrent_leader_conns_zero_rejected_with_valueerror() -> None:
    store = MemoryNodeStore(["localhost:9001"])
    with pytest.raises(ValueError, match="concurrent_leader_conns"):
        ClusterClient(store, concurrent_leader_conns=0)


@pytest.mark.asyncio
async def test_parallel_sweep_cancels_siblings_on_first_success() -> None:
    """When one probe wins, every other in-flight probe must receive
    ``CancelledError`` so its socket-drain runs and we don't leak FDs.
    """
    addresses = ["fast:9001", "slow-a:9001", "slow-b:9001"]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0)

    cancellations: dict[str, bool] = {}

    async def _query_leader(address: str, **_kw: object) -> str:
        if address == "fast:9001":
            # Yield once so siblings start running.
            await asyncio.sleep(0)
            return "fast:9001"
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            cancellations[address] = True
            raise
        return ""

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    leader = await cluster.find_leader()
    # Give cancelled siblings a chance to record their cancellation
    # via the gather inside the finally block.
    await asyncio.sleep(0)

    assert leader == "fast:9001"
    assert cancellations.get("slow-a:9001") is True
    assert cancellations.get("slow-b:9001") is True


@pytest.mark.asyncio
async def test_parallel_sweep_aggregate_error_carries_every_per_node_error() -> None:
    """All probes raise distinct transport errors. The final
    ``ClusterError`` message must name every node, and the
    ``__cause__`` must be a ``BaseExceptionGroup`` with one
    sub-exception per failed node — order non-deterministic but
    presence guaranteed."""
    addresses = ["node-a:9001", "node-b:9001", "node-c:9001"]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0)

    async def _query_leader(address: str, **_kw: object) -> None:
        raise DqliteConnectionError(f"refused on {address}")

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    with pytest.raises(ClusterError) as exc_info:
        await cluster.find_leader()

    msg = str(exc_info.value)
    for addr in addresses:
        assert addr in msg, f"address {addr} missing from aggregate error"

    cause = exc_info.value.__cause__
    assert isinstance(cause, BaseExceptionGroup)
    matched, _ = cause.split(DqliteConnectionError)
    assert matched is not None and len(matched.exceptions) == 3


@pytest.mark.asyncio
async def test_parallel_sweep_no_exceptions_returns_aggregate_no_leader_known() -> None:
    """All probes return ``None`` (no leader known). Final error
    raises with no ``__cause__`` — the message itself is the
    diagnostic. Pre-parallel behaviour preserved."""
    addresses = ["node-a:9001", "node-b:9001", "node-c:9001"]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0)

    cluster._query_leader = AsyncMock(return_value=None)

    with pytest.raises(ClusterError) as exc_info:
        await cluster.find_leader()

    assert exc_info.value.__cause__ is None
    msg = str(exc_info.value)
    for addr in addresses:
        assert addr in msg


@pytest.mark.asyncio
async def test_parallel_sweep_redirect_policy_propagates() -> None:
    """A redirect to a policy-rejected address propagates
    ``ClusterPolicyError`` out of the sweep, cancelling siblings."""
    addresses = ["node-a:9001", "node-b:9001"]
    store = MemoryNodeStore(addresses)

    def policy(address: str) -> bool:
        return False  # reject everything

    cluster = ClusterClient(store, timeout=10.0, redirect_policy=policy)

    async def _query_leader(address: str, **_kw: object) -> str:
        # Both nodes redirect to a forbidden target; whichever lands
        # first triggers the policy rejection.
        return "redirect-target:9001"

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()


@pytest.mark.asyncio
async def test_parallel_sweep_unknown_exception_propagates_after_sibling_cancel() -> None:
    """A programming-bug exception (TypeError) raised by a probe
    must propagate out of the sweep — the narrow except clauses do
    NOT catch it. Siblings are cancelled before the raise."""
    addresses = ["bug:9001", "slow:9001"]
    store = MemoryNodeStore(addresses)
    cluster = ClusterClient(store, timeout=10.0)

    cancellations: dict[str, bool] = {}

    async def _query_leader(address: str, **_kw: object) -> str:
        if address == "bug:9001":
            raise TypeError("programming bug")
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            cancellations[address] = True
            raise
        return ""

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    with pytest.raises(TypeError, match="programming bug"):
        await cluster.find_leader()

    await asyncio.sleep(0)
    assert cancellations.get("slow:9001") is True
