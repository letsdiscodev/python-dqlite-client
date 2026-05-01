"""Pin: ``ConnectionPool``'s ``min_size`` / ``max_size`` /
``max_attempts`` and ``ClusterClient.connect``'s ``max_attempts``
reject ``bool`` (``True`` / ``False``) — the only checks were
``< 0`` / ``< 1`` comparisons that silently coerced ``True`` to
``1`` and ``False`` to ``0``.

Mirrors the int/bool reject discipline applied across the wire
validators (``encode_int64``, ``Cursor.arraysize``,
``Header.__post_init__``).
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


def test_pool_min_size_rejects_bool() -> None:
    with pytest.raises(TypeError, match="min_size must be int"):
        ConnectionPool(addresses=["localhost:9001"], min_size=True)


def test_pool_max_size_rejects_bool() -> None:
    with pytest.raises(TypeError, match="max_size must be int"):
        ConnectionPool(addresses=["localhost:9001"], max_size=True)


def test_pool_max_attempts_rejects_bool() -> None:
    with pytest.raises(TypeError, match="max_attempts must be int"):
        ConnectionPool(
            addresses=["localhost:9001"],
            max_attempts=True,
        )


@pytest.mark.asyncio
async def test_cluster_connect_max_attempts_rejects_bool() -> None:
    cluster = ClusterClient(MemoryNodeStore())
    with pytest.raises(TypeError, match="max_attempts must be int"):
        await cluster.connect(database="x", max_attempts=True)
