"""describe/set_weight validate the address kwarg client-side before dialing, so a
malformed address raises ValueError at the call site (symmetric with add_node)."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
def client() -> ClusterClient:
    return ClusterClient(MemoryNodeStore(["127.0.0.1:9001"]))


@pytest.mark.parametrize(
    "bad_address",
    [
        "10.0.0.1: 9000",  # stray whitespace
        "user@host:9000",  # credentials-smuggle
        "10.0.0.1:0",  # zero port
        "10.0.0.1",  # missing port
    ],
)
async def test_describe_invalid_address_raises_at_call_site(
    client: ClusterClient, bad_address: str
) -> None:
    with pytest.raises(ValueError, match=r"describe: invalid address"):
        await client.describe(address=bad_address)


@pytest.mark.parametrize(
    "bad_address",
    [
        "10.0.0.1: 9000",
        "user@host:9000",
        "10.0.0.1:0",
        "10.0.0.1",
    ],
)
async def test_set_weight_invalid_address_raises_at_call_site(
    client: ClusterClient, bad_address: str
) -> None:
    with pytest.raises(ValueError, match=r"set_weight: invalid address"):
        await client.set_weight(1, address=bad_address)


async def test_add_node_invalid_address_wording_preserved(client: ClusterClient) -> None:
    """add_node's error wording stays symmetric: "<method>: invalid address"."""
    from dqlitewire import NodeRole

    with pytest.raises(ValueError, match=r"add_node: invalid address"):
        await client.add_node(2, "10.0.0.1: 9000", role=NodeRole.VOTER)
