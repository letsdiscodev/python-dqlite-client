"""Pin: ``MemoryNodeStore`` syntactically validates each address
via the same ``parse_address`` helper that ``_query_leader`` will
use later. Without this, malformed entries (unbracketed IPv6,
non-numeric port, IDN, credentials shape) leak ``ValueError``
through the ``find_leader`` sweep's narrow except tuple — aborting
the entire sweep with the wrong exception class.
"""

from __future__ import annotations

import pytest

from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


@pytest.mark.parametrize(
    "bad",
    [
        "no-port",
        "host:abc",
        "host:99999",
        "::1:9001",  # unbracketed IPv6
    ],
)
def test_init_rejects_malformed_address(bad: str) -> None:
    with pytest.raises(ValueError, match="not a valid"):
        MemoryNodeStore([bad])


def test_init_accepts_well_formed_addresses() -> None:
    store = MemoryNodeStore(["localhost:9001", "[::1]:9002"])
    nodes = list(store._nodes)
    assert len(nodes) == 2
    assert nodes[0].address == "localhost:9001"
    assert nodes[1].address == "[::1]:9002"


@pytest.mark.asyncio
async def test_set_nodes_rejects_malformed_address() -> None:
    store = MemoryNodeStore()
    with pytest.raises(ValueError, match="not a valid"):
        await store.set_nodes([NodeInfo(node_id=1, address="bogus:notaport", role=NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_set_nodes_accepts_well_formed_addresses() -> None:
    store = MemoryNodeStore()
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="localhost:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="[::1]:9002", role=NodeRole.STANDBY),
        ]
    )
    nodes = await store.get_nodes()
    assert len(nodes) == 2
