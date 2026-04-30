"""Pin: ``ClusterClient.find_leader``'s aggregate-of-per-node-errors
payload is capped at ``_MAX_AGGREGATE_ERROR_PAYLOAD`` so the final
``ClusterError`` does not grow O(N) in operator-configured node count.

The per-node snippet cap (``_MAX_ERROR_MESSAGE_SNIPPET = 200``)
already bounds M (per-node failure message size). Without the
aggregate cap, a 500-node store of failing peers produced ≥100 KB
held in the ClusterError args, in every traceback render, and in
every ``__cause__`` walk on a long-lived process's pool retry
loops.
"""

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

    huge = "x" * 50_000  # per-node hostile message size

    async def _fake_query(self: ClusterClient, address: str, **kwargs: object) -> str:
        raise DqliteConnectionError(huge)

    monkeypatch.setattr(ClusterClient, "_query_leader", _fake_query)

    with pytest.raises(ClusterError) as exc_info:
        await client.find_leader()

    aggregate = str(exc_info.value)
    # The error string is "Could not find leader. Errors: <joined>"
    # plus a small truncation marker. Bound the test loosely on the
    # aggregate cap plus a small fixed overhead (prefix +
    # truncation marker).
    assert len(aggregate) <= _MAX_AGGREGATE_ERROR_PAYLOAD + 256, (
        f"Aggregate error payload {len(aggregate)} exceeds "
        f"_MAX_AGGREGATE_ERROR_PAYLOAD {_MAX_AGGREGATE_ERROR_PAYLOAD} "
        f"+ overhead. The N axis (configured node-store size) is "
        f"operator-controlled and unbounded — without the aggregate "
        f"cap, a 500-node store of failing peers produced >100 KB."
    )
    # Sanity: the truncation marker should be present (the test
    # configuration is designed to exceed the cap).
    assert "[aggregate truncated" in aggregate
