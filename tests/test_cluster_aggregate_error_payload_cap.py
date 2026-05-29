"""find_leader's aggregate per-node error payload is capped so ClusterError
does not grow O(N) in operator-configured node count."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import (
    _MAX_AGGREGATE_ERROR_PAYLOAD,
    ClusterClient,
)
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_find_leader_aggregate_error_payload_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addrs = [f"10.0.{i // 256}.{i % 256}:9001" for i in range(500)]
    store = MemoryNodeStore(addrs)
    client = ClusterClient(store, timeout=0.01)

    huge = "x" * 50_000

    async def _fake_query(self: ClusterClient, address: str, **kwargs: object) -> str:
        raise DqliteConnectionError(huge)

    monkeypatch.setattr(ClusterClient, "_query_leader", _fake_query)

    with pytest.raises(ClusterError) as exc_info:
        await client.find_leader()

    aggregate = str(exc_info.value)
    # Bound on the aggregate cap plus fixed overhead (prefix + truncation marker).
    assert len(aggregate) <= _MAX_AGGREGATE_ERROR_PAYLOAD + 256, (
        f"Aggregate error payload {len(aggregate)} exceeds "
        f"_MAX_AGGREGATE_ERROR_PAYLOAD {_MAX_AGGREGATE_ERROR_PAYLOAD} "
        f"+ overhead. The N axis (configured node-store size) is "
        f"operator-controlled and unbounded — without the aggregate "
        f"cap, a 500-node store of failing peers produced >100 KB."
    )
    assert "[aggregate truncated" in aggregate
