"""``MemoryNodeStore.set_nodes`` serialises concurrent writers
via an asyncio.Lock so neither caller's update is silently lost
(last-writer-wins). Mirrors the discipline ``YamlNodeStore``
already applies.
"""

import asyncio

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_concurrent_set_nodes_serialised() -> None:
    """Two concurrent set_nodes calls do not race on the final
    tuple assignment. Either A's update or B's update wins, but
    the final state is one of the two complete sets — never a
    half-applied mix.
    """
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

    # Fire both writers concurrently.
    await asyncio.gather(store.set_nodes(set_a), store.set_nodes(set_b))

    final = await store.get_nodes()
    addrs = {n.address for n in final}
    # The result must be exactly one of the two sets — not a mix.
    set_a_addrs = {n.address for n in set_a}
    set_b_addrs = {n.address for n in set_b}
    assert addrs in (set_a_addrs, set_b_addrs)


@pytest.mark.asyncio
async def test_set_nodes_lock_eager_constructed() -> None:
    """The lock is constructed eagerly in __init__ — asyncio.Lock
    is loop-agnostic at construction time on Python 3.10+ (binds
    to the loop on first acquire). Lazy construction would lose
    the mutual-exclusion contract under concurrent first-time
    callers (each builds its own private Lock)."""
    store = MemoryNodeStore(["seed:9001"])
    assert isinstance(store._set_nodes_lock, asyncio.Lock)
    await store.set_nodes([NodeInfo(1, "a:9001", NodeRole.VOTER)])
    assert isinstance(store._set_nodes_lock, asyncio.Lock)
