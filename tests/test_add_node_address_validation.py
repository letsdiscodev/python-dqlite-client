"""Pin: ``ClusterClient.add_node`` rejects malformed addresses at the
call site via the in-tree strict ``parse_address`` parser.

Previously ``add_node`` only checked ``isinstance(address, str) and
address``, so non-numeric ports, unbracketed IPv6, whitespace, CRLF,
credentials-smuggle ``user@host``, etc. were forwarded to the server
and surfaced asynchronously as connection failures. This pin ensures
the rejection lands at the call site with a specific diagnostic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqlitewire.constants import NodeRole


def _make_cluster_client_skeleton() -> ClusterClient:
    """Construct a ClusterClient that won't actually connect — the
    address validation runs BEFORE any network I/O, so a no-op store
    suffices for these synthetic-input tests."""
    from dqliteclient.node_store import MemoryNodeStore

    store = MemoryNodeStore([])
    client = ClusterClient(store, dial_timeout=0.5, attempt_timeout=0.5)
    # Skip find_leader by patching the leader cache; the validation
    # check fires before find_leader is invoked.
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
    """Each malformed address shape is rejected at the call site
    with ValueError from parse_address."""
    client = _make_cluster_client_skeleton()
    with pytest.raises(ValueError, match="add_node|invalid"):
        await client.add_node(node_id=99, address=bad_address, role=NodeRole.VOTER)


@pytest.mark.asyncio
async def test_add_node_empty_string_rejected() -> None:
    """Negative pin: empty string is rejected by the existing
    ``isinstance/empty`` check, before parse_address runs."""
    client = _make_cluster_client_skeleton()
    with pytest.raises(TypeError, match="non-empty"):
        await client.add_node(node_id=99, address="", role=NodeRole.VOTER)
