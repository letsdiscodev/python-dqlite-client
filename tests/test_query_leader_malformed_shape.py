"""``_query_leader`` must reject BOTH wire-illegal LeaderResponse
shapes, not just ``(node_id!=0, address="")``. Upstream dqlite sets
id and address atomically; the mirror shape ``(node_id=0,
address!="")`` is an invariant violation that the client must raise
as ProtocolError rather than treating as a redirect.

For the ``(0, "peer:9000")`` arm we hand-build the wire bytes
directly. The Python ``LeaderResponse.encode_body`` rejects that
shape at encode time (raft-leader atomicity invariant — see
``test_leader_response_encode_side_atomicity.py`` in dqlite-wire),
so simulating a malicious / non-conforming peer requires assembling
the bytes without going through the Python constructor's safety
checks. This matches the real-world threat model: a hostile peer
controlling the wire bytes is exactly the path the decode-side
reject defends against.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


def _hand_build_leader_response_bytes(node_id: int, address: str) -> bytes:
    """Build wire bytes for a LeaderResponse WITHOUT the Python
    constructor's encode-side atomicity check. Mirrors what a hostile
    peer would emit."""
    from dqlitewire.constants import ResponseType
    from dqlitewire.messages.base import Header
    from dqlitewire.types import encode_text, encode_uint64

    body = encode_uint64(node_id) + encode_text(address)
    header = Header(size_words=len(body) // 8, msg_type=ResponseType.LEADER, schema=0)
    return header.encode() + body


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

    from dqlitewire.messages import WelcomeResponse

    # The ``(1, "")`` arm can go through the normal encoder (the wire
    # check only rejects ``(0, non-empty)``); the ``(0, "peer:9000")``
    # arm needs hand-built bytes.
    if node_id == 0 and address_str:
        leader_bytes = _hand_build_leader_response_bytes(node_id, address_str)
    else:
        from dqlitewire.messages import LeaderResponse

        leader_bytes = LeaderResponse(node_id=node_id, address=address_str).encode()

    mock_reader.read.side_effect = [
        WelcomeResponse(heartbeat_timeout=15000).encode(),
        leader_bytes,
    ]

    with (
        patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
        # ProtocolError propagates through find_leader as ClusterError
        # (the outer retry loop wraps the per-node errors).
        pytest.raises(ClusterError),
    ):
        await client.find_leader()
