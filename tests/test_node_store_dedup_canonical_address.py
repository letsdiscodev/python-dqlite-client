"""``_validate_and_normalise_nodes`` dedups by the ``parse_address`` canonical (host, port)
tuple, not the lexical string, so case/format variants don't make find_leader double-probe."""

from __future__ import annotations

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_dedups_case_insensitive_hostname() -> None:
    """Mixed-case hostname variants are one canonical address (RFC 1035); keep one entry."""
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="Node1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="node1:9001", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="node2:9001", role=NodeRole.SPARE),
        ]
    )
    nodes = await store.get_nodes()
    # First-wins dedup: Node1 kept, the second mixed-case variant dropped.
    assert len(nodes) == 2, (
        f"expected 2 nodes after canonical dedup of Node1:9001 vs node1:9001, "
        f"got {len(nodes)}: {[n.address for n in nodes]}"
    )
    addresses = {n.address.lower() for n in nodes}
    assert "node1:9001" in addresses
    assert "node2:9001" in addresses


@pytest.mark.asyncio
async def test_set_nodes_dedups_ipv6_short_vs_long_form() -> None:
    """IPv6 short and long forms parse to the same canonical tuple; only one survives."""
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="[::1]:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="[0:0:0:0:0:0:0:1]:9001", role=NodeRole.STANDBY),
        ]
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 1, (
        f"expected 1 node after IPv6 short-vs-long-form dedup, got {len(nodes)}: "
        f"{[n.address for n in nodes]}"
    )
