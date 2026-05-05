"""Pin: ``MemoryNodeStore.set_nodes`` serialises concurrent
callers, so neither caller's final tuple-assignment is lost.

Before the fix, two tasks racing on a freshly-constructed store
would each pass the ``if self._set_nodes_lock is None`` check on
the same scheduling slice, each construct a distinct
``asyncio.Lock``, and acquire their own private lock — neither
saw the other. The final ``self._nodes = tuple(unique)`` writes
interleaved and one update was lost.
"""

import asyncio

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire.constants import NodeRole


def _ni(address: str) -> NodeInfo:
    return NodeInfo(node_id=1, address=address, role=NodeRole.VOTER)


@pytest.mark.asyncio
async def test_concurrent_first_set_nodes_serialised() -> None:
    """Two tasks call set_nodes simultaneously on a fresh store.
    Both must successfully complete; the final state must reflect
    one of the two updates exactly (no torn / interleaved write)."""
    store = MemoryNodeStore(["seed:9001"])

    a_nodes = [_ni("a:9001"), _ni("a:9002"), _ni("a:9003")]
    b_nodes = [_ni("b:9001"), _ni("b:9002"), _ni("b:9003")]

    async def update(nodes: list[NodeInfo]) -> None:
        await store.set_nodes(nodes)

    await asyncio.gather(update(a_nodes), update(b_nodes))

    final = list(await store.get_nodes())
    addresses = [n.address for n in final]
    # Either A's update or B's — not a mix of both.
    assert addresses == ["a:9001", "a:9002", "a:9003"] or addresses == [
        "b:9001",
        "b:9002",
        "b:9003",
    ], f"torn write: {addresses}"


@pytest.mark.asyncio
async def test_set_nodes_lock_eager_constructed() -> None:
    """Pin the constructor invariant: the lock is constructed
    eagerly in __init__, not lazily on first set_nodes call.
    Without eager construction, two first-time callers each
    construct their own private Lock and the mutex contract is
    broken on the very first contended call."""
    store = MemoryNodeStore(["seed:9001"])
    assert isinstance(store._set_nodes_lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_many_concurrent_set_nodes_one_winner() -> None:
    """Stress: 16 concurrent set_nodes; final state matches
    exactly one of the 16 inputs."""
    store = MemoryNodeStore(["seed:9001"])
    inputs = [[_ni(f"node-{i}:9001")] for i in range(16)]

    await asyncio.gather(*(store.set_nodes(inp) for inp in inputs))

    final = list(await store.get_nodes())
    assert len(final) == 1
    matching = [inp for inp in inputs if [n.address for n in final] == [n.address for n in inp]]
    assert len(matching) == 1
