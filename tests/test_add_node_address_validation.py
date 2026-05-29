"""Pin: ``ClusterClient.add_node`` rejects malformed addresses at the call site
(via strict ``parse_address``) rather than forwarding them to the server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqlitewire.constants import NodeRole


def _make_cluster_client_skeleton() -> ClusterClient:
    """A ClusterClient that never connects; address validation runs before I/O."""
    from dqliteclient.node_store import MemoryNodeStore

    store = MemoryNodeStore([])
    client = ClusterClient(store, dial_timeout=0.5, attempt_timeout=0.5)
    client._set_last_known_leader = MagicMock()
    return client


@pytest.mark.parametrize(
    "bad_address",
    [
        "host:abc",  # non-numeric port
        "host:99999",  # port out of range
        "host:0",  # port 0
        "host: 9001",  # whitespace in port
        " host:9001",  # leading whitespace
        "host:9001\r\n",  # trailing CRLF
        "user@host:9001",  # credentials smuggle
        "host\nname:9001",  # embedded newline
        ":9001",  # empty host
        "host:",  # empty port
        "::1:9001",  # unbracketed IPv6
    ],
)
@pytest.mark.asyncio
async def test_add_node_rejects_malformed_address(bad_address: str) -> None:
    """Each malformed address is rejected with ValueError from parse_address."""
    client = _make_cluster_client_skeleton()
    with pytest.raises(ValueError, match="add_node|invalid"):
        await client.add_node(node_id=99, address=bad_address, role=NodeRole.VOTER)


@pytest.mark.asyncio
async def test_add_node_empty_string_rejected() -> None:
    """Empty string is rejected by the isinstance/empty check before parse_address."""
    client = _make_cluster_client_skeleton()
    with pytest.raises(TypeError, match="non-empty"):
        await client.add_node(node_id=99, address="", role=NodeRole.VOTER)
