"""Pin: ``ClusterClient._find_leader_impl`` chains per-node
failures via ``BaseExceptionGroup`` when more than one node
contributed a real exception.

Pre-fix, only the LAST iteration's exception was preserved on
``__cause__`` — code that branches on the cause class (e.g.
routing security alerts for ``ProtocolError``-from-malformed-
redirect) saw a non-deterministic decision based on iteration
ordering.

Mirrors ``ConnectionPool.initialize``'s
``BaseExceptionGroup``-on-multi-failure discipline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterError,
    DqliteConnectionError,
    ProtocolError,
)
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_find_leader_aggregate_chains_via_exception_group_when_multiple_failures() -> None:
    store = MemoryNodeStore(["node-a:9001", "node-b:9001", "node-c:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    # node-a → ProtocolError (security-relevant)
    # node-b → DqliteConnectionError (transport)
    # node-c → TimeoutError
    async def _query_leader_per_node(address: str, **_kw: object) -> None:
        if address == "node-a:9001":
            raise ProtocolError("malformed redirect from node-a")
        if address == "node-b:9001":
            raise DqliteConnectionError("connection refused on node-b")
        raise TimeoutError("node-c timed out")

    cluster._query_leader = AsyncMock(side_effect=_query_leader_per_node)

    with pytest.raises(ClusterError) as exc_info:
        await cluster.find_leader()

    cause = exc_info.value.__cause__
    # Multi-failure case: cause must be BaseExceptionGroup so callers
    # branching on .split(ProtocolError) recover the security-relevant
    # exception regardless of iteration order.
    assert isinstance(cause, BaseExceptionGroup), (
        f"expected BaseExceptionGroup chain on multi-failure, got {type(cause).__name__}"
    )
    matched, _ = cause.split(ProtocolError)
    assert matched is not None and len(matched.exceptions) == 1
    matched_transport, _ = cause.split(DqliteConnectionError)
    assert matched_transport is not None and len(matched_transport.exceptions) == 1


@pytest.mark.asyncio
async def test_find_leader_aggregate_keeps_narrow_chain_on_single_failure() -> None:
    """Backward-compat: single-failure case keeps the narrow chain so
    callers that branch on ``e.__cause__`` type continue to work."""
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=0.5)

    async def _query_leader_fails(address: str, **_kw: object) -> None:
        raise ProtocolError("malformed redirect")

    cluster._query_leader = AsyncMock(side_effect=_query_leader_fails)

    with pytest.raises(ClusterError) as exc_info:
        await cluster.find_leader()

    cause = exc_info.value.__cause__
    assert isinstance(cause, ProtocolError)
    assert "malformed redirect" in str(cause)
