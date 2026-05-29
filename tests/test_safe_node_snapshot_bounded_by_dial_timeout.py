"""``_safe_node_snapshot`` bounds the ``NodeStore.get_nodes`` await by ``dial_timeout``;
otherwise a blocking third-party store (etcd/consul) would hang ``find_leader`` forever,
since the outer envelope and per-probe timeouts both run AFTER the store snapshot."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import NodeInfo

pytestmark = pytest.mark.asyncio


class _SlowNodeStore:
    """A NodeStore whose ``get_nodes`` blocks for 60 s."""

    async def get_nodes(self) -> Sequence[NodeInfo]:
        await asyncio.sleep(60)
        return ()

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        return None


async def test_slow_node_store_get_nodes_bounded_by_dial_timeout() -> None:
    store = _SlowNodeStore()
    client = ClusterClient(store, dial_timeout=0.1)
    start = time.monotonic()
    with pytest.raises((TimeoutError, DqliteConnectionError)):
        await client.find_leader()
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, (
        f"_safe_node_snapshot must bound the get_nodes call by dial_timeout; "
        f"elapsed {elapsed:.2f}s (expected ~0.1s)"
    )
