"""``ClusterClient.open_admin_connection`` forwards
``trust_server_heartbeat``, ``max_total_rows``, and
``max_continuation_frames`` to the underlying ``DqliteProtocol``
construction. Previously every admin RPC (cluster_info, dump,
transfer_leadership, ...) ran with the dqlitewire defaults regardless
of how the operator configured the parent ``ClusterClient``.

``trust_server_heartbeat`` is security-relevant (opt-in widening of
the per-read deadline). ``max_total_rows`` and
``max_continuation_frames`` matter for ``dump`` paths whose
multi-GB results would blow through the default frame cap.

Symmetric forwarding into ``_query_leader`` is also pinned here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster_with_governors() -> ClusterClient:
    return ClusterClient(
        MemoryNodeStore(["localhost:9001"]),
        timeout=2.0,
        max_total_rows=12_345,
        max_continuation_frames=678,
        trust_server_heartbeat=True,
    )


@pytest.mark.asyncio
async def test_open_admin_connection_forwards_trust_server_heartbeat() -> None:
    cluster = _make_cluster_with_governors()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object):
        return reader, writer

    captured_kwargs: dict[str, object] = {}

    def capture_protocol(*args: object, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        return fake_proto

    with (
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", side_effect=capture_protocol),
    ):
        async with cluster.open_admin_connection("localhost:9001"):
            pass

    assert captured_kwargs.get("trust_server_heartbeat") is True
    assert captured_kwargs.get("max_total_rows") == 12_345
    assert captured_kwargs.get("max_continuation_frames") == 678


@pytest.mark.asyncio
async def test_query_leader_forwards_governors() -> None:
    cluster = _make_cluster_with_governors()
    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.get_leader = AsyncMock(return_value=(1, "localhost:9001"))
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open(host: str, port: int, **_kwargs: object):
        return reader, writer

    captured_kwargs: dict[str, object] = {}

    def capture_protocol(*args: object, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        return fake_proto

    with (
        patch("dqliteclient.cluster.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", side_effect=capture_protocol),
    ):
        await cluster._query_leader("localhost:9001", trust_server_heartbeat=True)

    assert captured_kwargs.get("max_total_rows") == 12_345
    assert captured_kwargs.get("max_continuation_frames") == 678


def test_cluster_client_governor_defaults_match_wire_defaults() -> None:
    """Default-constructed ClusterClient picks up the dqlitewire
    DEFAULT_MAX_TOTAL_ROWS / DEFAULT_MAX_CONTINUATION_FRAMES — the
    forwarding does not silently downgrade from those defaults."""
    from dqlitewire import (
        DEFAULT_MAX_CONTINUATION_FRAMES,
        DEFAULT_MAX_TOTAL_ROWS,
    )

    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]))
    assert cluster._max_total_rows == DEFAULT_MAX_TOTAL_ROWS
    assert cluster._max_continuation_frames == DEFAULT_MAX_CONTINUATION_FRAMES
    assert cluster._trust_server_heartbeat is False
