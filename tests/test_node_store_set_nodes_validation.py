"""Pin: ``MemoryNodeStore.set_nodes`` mirrors ``__init__``'s
strip / dedup / empty-rejection validation contract.

Cycle 22 applied the ``__init__`` validation to
``set_nodes`` to close a runtime-update bypass. Without
these tests, a regression that drops the validation block
silently re-introduces the exact defect the cycle-22
commit was meant to prevent.

Six contract behaviours covered:

* ``TypeError`` on non-string address.
* ``ValueError`` on empty/whitespace-only address.
* Whitespace stripping with NodeInfo rebuild (frozen
  dataclass).
* Dedup of duplicate addresses (first wins).
* Stripped variants of an already-seen address are dedup'd.
"""

from __future__ import annotations

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_rejects_non_string_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(TypeError, match="(?i)address must be"):
        await store.set_nodes(
            [NodeInfo(node_id=1, address=12345, role=NodeRole.VOTER)]  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_set_nodes_rejects_empty_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(ValueError, match="(?i)non-empty"):
        await store.set_nodes([NodeInfo(node_id=1, address="", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_rejects_whitespace_only_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(ValueError, match="(?i)non-empty"):
        await store.set_nodes([NodeInfo(node_id=1, address="   ", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_strips_whitespace() -> None:
    store = MemoryNodeStore()
    await store.set_nodes([NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER)])
    nodes = await store.get_nodes()
    assert nodes[0].address == "127.0.0.1:9001"


@pytest.mark.asyncio
async def test_set_nodes_dedups_duplicates_first_wins() -> None:
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="127.0.0.1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="  127.0.0.1:9001  ", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="127.0.0.1:9002", role=NodeRole.SPARE),
        ]
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 2
    assert nodes[0].node_id == 1
    assert nodes[1].node_id == 3


@pytest.mark.asyncio
async def test_set_nodes_rebuilds_nodeinfo_when_address_stripped() -> None:
    """Frozen-dataclass rebuild: a stripped address yields a new
    NodeInfo instance, not the caller's original. A refactor that
    drops the rebuild branch (and tries in-place mutation, which
    would TypeError on the frozen dataclass) surfaces here."""
    store = MemoryNodeStore()
    original = NodeInfo(node_id=1, address="  127.0.0.1:9001  ", role=NodeRole.VOTER)
    await store.set_nodes([original])
    nodes = await store.get_nodes()
    stored = nodes[0]
    assert stored is not original
    assert stored.address == "127.0.0.1:9001"
    # node_id and role are preserved across the rebuild.
    assert stored.node_id == original.node_id
    assert stored.role == original.role
