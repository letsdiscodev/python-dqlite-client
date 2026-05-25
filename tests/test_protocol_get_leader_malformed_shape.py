"""Pin: ``DqliteProtocol.get_leader`` rejects the wire-illegal
``(node_id=0, address!="")`` shape with a clear ``ProtocolError``.

The cluster-layer wrappers (``_query_leader``, ``leader_info``) used
to be the only enforcers of the upstream ``raft_leader`` invariant
("id and address are paired; both filled or both zero/NULL"). A
third-party caller wiring ``DqliteProtocol.get_leader`` directly
into a custom probe / monitor would get the malformed pair verbatim
and could trust ``(0, attacker-addr)`` as a leader hint.

The current wire decoder already rejects ``(0, nonempty)`` from
the modern wire path (``LeaderResponse.decode_body``), so the
end-to-end behaviour is already correct. The wire-layer guard
in ``get_leader`` is defence-in-depth: if the wire check is ever
relaxed (a future legacy-format path, a refactor, etc.) the
protocol layer still raises before handing the malformed pair to
the caller.

The mirror ``(N, "")`` "no leader known" shape is deliberately NOT
raised here — cluster wrappers want the per-address ``RAFT_NOMEM``
breadcrumb the protocol layer can't synthesise.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol


@pytest.mark.asyncio
async def test_get_leader_defence_in_depth_rejects_node_id_zero_with_address() -> None:
    """The wire decoder already rejects ``(0, addr)``; this test
    bypasses the decoder and feeds a ``LeaderResponse(0, addr)``
    directly to ``get_leader`` to exercise the protocol-layer
    defence-in-depth guard."""
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

    # Bypass the wire decode entirely: have ``_read_response`` return
    # a LeaderResponse instance with the malformed shape (which the
    # legacy-decode path can construct legitimately —
    # ``decode_body_legacy`` hard-codes ``node_id=0`` and reads the
    # address from the legacy wire body).
    malformed = LeaderResponse(node_id=0, address="attacker:9000")
    with (
        patch.object(protocol, "_send", new=AsyncMock()),
        patch.object(protocol, "_read_response", new=AsyncMock(return_value=malformed)),
        pytest.raises(ProtocolError, match="expected both or neither"),
    ):
        await protocol.get_leader()


@pytest.mark.asyncio
async def test_get_leader_passes_through_no_leader_known_shape() -> None:
    """The ``(0, "")`` "no leader known" shape is passed through
    verbatim — callers normalise it."""
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
    """The ``(N, "")`` ``RAFT_NOMEM`` transient stays at the wire
    layer for cluster-layer normalisation."""
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
    """Sanity: the normal ``(N, addr)`` pair is returned verbatim."""
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
