"""Pin: ``ClusterClient._safe_node_snapshot`` wraps the
``NodeStore.get_nodes`` await in an ``asyncio.timeout`` cancel scope
bounded by ``dial_timeout``.

In-tree stores (``MemoryNodeStore``, ``YamlNodeStore``) return
synchronously so the wrap is a no-op for them. A third-party store
backed by blocking I/O (etcd / consul / service-discovery lookup)
would otherwise hang ``find_leader`` indefinitely because no
``asyncio.timeout`` envelope previously wrapped the store call. The
outer ``find_leader`` envelope and the per-probe timeouts both run
AFTER the store snapshot — neither bounds it.

Mirrors go-dqlite's ``NodeStore.Get(ctx)`` contract and the
``asyncio.timeout`` discipline applied to every other awaitable in
``cluster.py``.
"""

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
    """Test fixture: a NodeStore whose ``get_nodes`` blocks for 60 s.

    Without the timeout wrap, ``find_leader`` hangs until the outer
    deadline (none here) fires. With the wrap, the store call is
    bounded by ``dial_timeout`` and surfaces a TimeoutError-class
    raise within ``dial_timeout``.
    """

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
