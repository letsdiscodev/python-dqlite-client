"""Pin: ``_validate_and_normalise_nodes`` accepts generator inputs.

A perfectly idiomatic call-site like::

    await store.set_nodes(NodeInfo(...) for n in seeds)

used to trip a bare ``TypeError: object of type 'generator' has no
len()`` deep inside the helper because the first runtime use was an
upfront ``len(nodes) > _WIRE_MAX_NODES`` cap check. The helper now
materialises non-list/tuple inputs first so that:

* Generators succeed (DX win).
* The wire-cap check still fires after materialisation.
"""

from __future__ import annotations

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_accepts_generator_expression() -> None:
    store = MemoryNodeStore()
    nodes_gen = (
        NodeInfo(node_id=i + 1, address=f"127.0.0.1:{9001 + i}", role=NodeRole.VOTER)
        for i in range(3)
    )
    await store.set_nodes(nodes_gen)  # type: ignore[arg-type]
    nodes = await store.get_nodes()
    assert len(nodes) == 3
    assert [n.node_id for n in nodes] == [1, 2, 3]


@pytest.mark.asyncio
async def test_set_nodes_generator_still_subject_to_wire_cap() -> None:
    from dqlitewire.messages.responses import _MAX_NODE_COUNT as wire_max

    store = MemoryNodeStore()
    over_cap = wire_max + 1
    nodes_gen = (
        NodeInfo(node_id=i + 1, address=f"127.0.0.1:{1000 + i}", role=NodeRole.VOTER)
        for i in range(over_cap)
    )
    with pytest.raises(ValueError, match="(?i)too many nodes"):
        await store.set_nodes(nodes_gen)  # type: ignore[arg-type]
