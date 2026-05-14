"""Pin: ``_validate_and_normalise_nodes`` deduplicates by the
``parse_address`` canonical tuple, not by the stripped lexical
address.

Pre-fix the dedup key was the stripped raw address string; entries
like ``NodeInfo(address="Node1:9001")`` and
``NodeInfo(address="node1:9001")`` both survived because their
lexical strings differ, even though ``parse_address`` returns the
same ``(host, port)`` tuple (RFC 1035 §2.3.3 case-insensitive
hostname). ``find_leader`` then probes the same peer twice, halving
the per-attempt probe budget.

Mirrors the canonical-tuple discipline already established in
``_addr_equiv`` at ``cluster.py:174-190``.
"""

from __future__ import annotations

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_set_nodes_dedups_case_insensitive_hostname() -> None:
    """Mixed-case hostname variants are the same canonical address
    per RFC 1035; the store must retain only one entry."""
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="Node1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="node1:9001", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="node2:9001", role=NodeRole.SPARE),
        ]
    )
    nodes = await store.get_nodes()
    # First-wins dedup: Node1 kept, the second mixed-case variant of
    # the same canonical host dropped, node2 retained as distinct.
    assert len(nodes) == 2, (
        f"expected 2 nodes after canonical dedup of Node1:9001 vs node1:9001, "
        f"got {len(nodes)}: {[n.address for n in nodes]}"
    )
    addresses = {n.address.lower() for n in nodes}
    assert "node1:9001" in addresses
    assert "node2:9001" in addresses


@pytest.mark.asyncio
async def test_set_nodes_dedups_ipv6_short_vs_long_form() -> None:
    """``[::1]:9001`` and ``[0:0:0:0:0:0:0:1]:9001`` parse to the same
    canonical ``(host, port)`` tuple — only one should survive dedup."""
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
