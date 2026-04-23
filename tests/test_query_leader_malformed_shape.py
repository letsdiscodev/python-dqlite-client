"""``_query_leader`` must reject BOTH wire-illegal LeaderResponse
shapes, not just ``(node_id!=0, address="")``. Upstream dqlite sets
id and address atomically; the mirror shape ``(node_id=0,
address!="")`` is an invariant violation that the client must raise
as ProtocolError rather than treating as a redirect.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.parametrize(
    "node_id,address_str",
    [
        (1, ""),  # previously-covered arm
        (0, "peer:9000"),  # mirror arm — must also raise
    ],
)
@pytest.mark.asyncio
async def test_query_leader_rejects_both_wire_illegal_shapes(
    node_id: int, address_str: str
) -> None:
    store = MemoryNodeStore(["localhost:9001"])
    client = ClusterClient(store, timeout=1.0)

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    from dqlitewire.messages import LeaderResponse, WelcomeResponse

    mock_reader.read.side_effect = [
        WelcomeResponse(heartbeat_timeout=15000).encode(),
        LeaderResponse(node_id=node_id, address=address_str).encode(),
    ]

    with (
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
        # ProtocolError propagates through find_leader as ClusterError
        # (the outer retry loop wraps the per-node errors).
        pytest.raises(ClusterError),
    ):
        await client.find_leader()
