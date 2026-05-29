"""Pin: ``DqliteProtocol.get_leader`` rejects the wire-illegal
``(node_id=0, address!="")`` shape with a clear ``ProtocolError``
(defence-in-depth; the wire decoder already rejects it). The mirror
``(N, "")`` shape is passed through for cluster-layer normalisation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol


@pytest.mark.asyncio
async def test_get_leader_defence_in_depth_rejects_node_id_zero_with_address() -> None:
    """Feed ``LeaderResponse(0, addr)`` past the decoder to exercise
    the protocol-layer defence-in-depth guard."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    malformed = LeaderResponse(node_id=0, address="attacker:9000")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=malformed)),
        pytest.raises(ProtocolError, match="expected both or neither"),
    ):
        await protocol.get_leader()


@pytest.mark.asyncio
async def test_get_leader_passes_through_no_leader_known_shape() -> None:
    """The ``(0, "")`` "no leader known" shape is passed through verbatim."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    no_leader = LeaderResponse(node_id=0, address="")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=no_leader)),
    ):
        node_id, address = await protocol.get_leader()
    assert node_id == 0
    assert address == ""


@pytest.mark.asyncio
async def test_get_leader_passes_through_raft_nomem_transient_shape() -> None:
    """The ``(N, "")`` ``RAFT_NOMEM`` transient is passed through."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    raft_nomem = LeaderResponse(node_id=5, address="")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=raft_nomem)),
    ):
        node_id, address = await protocol.get_leader()
    assert node_id == 5
    assert address == ""


@pytest.mark.asyncio
async def test_get_leader_happy_path_unchanged() -> None:
    """The normal ``(N, addr)`` pair is returned verbatim."""
    from dqlitewire.messages import LeaderResponse

    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.drain = AsyncMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    protocol = DqliteProtocol(
        mock_reader,
        mock_writer,
        timeout=1.0,
        address="peer-host:9000",
    )

    happy = LeaderResponse(node_id=2, address="real-leader:9001")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=happy)),
    ):
        node_id, address = await protocol.get_leader()
    assert node_id == 2
    assert address == "real-leader:9001"
