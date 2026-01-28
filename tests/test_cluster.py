"""Tests for cluster management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo


class TestClusterClient:
    def test_from_addresses(self) -> None:
        client = ClusterClient.from_addresses(["localhost:9001", "localhost:9002"])
        assert client._timeout == 10.0

    async def test_find_leader_no_nodes(self) -> None:
        store = MemoryNodeStore()
        client = ClusterClient(store)

        with pytest.raises(ClusterError, match="No nodes configured"):
            await client.find_leader()

    async def test_find_leader_all_unreachable(self) -> None:
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        client = ClusterClient(store, timeout=0.1)

        with (
            patch(
                "asyncio.open_connection",
                side_effect=OSError("Connection refused"),
            ),
            pytest.raises(ClusterError, match="Could not find leader"),
        ):
            await client.find_leader()

    async def test_find_leader_success(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import LeaderResponse, WelcomeResponse

        # First call for handshake, second for leader query
        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            LeaderResponse(node_id=1, address="").encode(),  # Empty = this node is leader
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            leader = await client.find_leader()

        assert leader == "localhost:9001"

    async def test_find_leader_redirect(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import LeaderResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            LeaderResponse(node_id=2, address="localhost:9002").encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            leader = await client.find_leader()

        assert leader == "localhost:9002"

    async def test_update_nodes(self) -> None:
        store = MemoryNodeStore()
        client = ClusterClient(store)

        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=1),
            NodeInfo(node_id=2, address="node2:9002", role=2),
        ]
        await client.update_nodes(nodes)

        stored = await store.get_nodes()
        assert len(stored) == 2
