"""``MemoryNodeStore.set_nodes`` serialises concurrent writers so no update is lost."""

import asyncio

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_concurrent_set_nodes_serialised() -> None:
    """Final state is one of the two complete sets, never a half-applied mix."""
    store = MemoryNodeStore()
    set_a = [
        NodeInfo(node_id=1, address="a:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=2, address="b:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=3, address="c:9001", role=NodeRole.VOTER),
    ]
    set_b = [
        NodeInfo(node_id=4, address="d:9001", role=NodeRole.VOTER),
        NodeInfo(node_id=5, address="e:9001", role=NodeRole.VOTER),
    ]

    await asyncio.gather(store.set_nodes(set_a), store.set_nodes(set_b))

    final = await store.get_nodes()
    addrs = {n.address for n in final}
    set_a_addrs = {n.address for n in set_a}
    set_b_addrs = {n.address for n in set_b}
    assert addrs in (set_a_addrs, set_b_addrs)


@pytest.mark.asyncio
async def test_set_nodes_lock_eager_constructed() -> None:
    """The lock is constructed eagerly in __init__; asyncio.Lock binds to the loop on
    first acquire, so lazy construction would break mutual exclusion for first-time callers."""
    store = MemoryNodeStore(["seed:9001"])
    assert isinstance(store._set_nodes_lock, asyncio.Lock)
    await store.set_nodes([NodeInfo(1, "a:9001", NodeRole.VOTER)])
    assert isinstance(store._set_nodes_lock, asyncio.Lock)
