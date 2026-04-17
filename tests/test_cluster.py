"""Tests for cluster management."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError, DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo


class TestClusterClient:
    def test_from_addresses(self) -> None:
        client = ClusterClient.from_addresses(["localhost:9001", "localhost:9002"])
        assert client._timeout == 10.0

    def test_zero_timeout_raises(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        with pytest.raises(ValueError, match="timeout must be positive"):
            ClusterClient(store, timeout=0)

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

    async def test_find_leader_no_leader_known(self) -> None:
        """node_id=0 with empty address means no leader known, not 'this is the leader'."""
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
            LeaderResponse(node_id=0, address="").encode(),  # No leader known
        ]
        mock_reader.read.side_effect = responses

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            pytest.raises(ClusterError, match="Could not find leader"),
        ):
            await client.find_leader()

    async def test_find_leader_node_hangs_after_connect(self) -> None:
        """A node that accepts TCP but hangs on handshake should be timed out."""
        import asyncio

        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=0.1)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        # Server accepts connection but never sends a response
        async def hang_forever(*args, **kwargs):
            await asyncio.sleep(100)
            return b""

        mock_reader.read.side_effect = hang_forever

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            pytest.raises(ClusterError, match="Could not find leader"),
        ):
            await client.find_leader()

    async def test_query_leader_closes_writer_on_handshake_error(self) -> None:
        """Writer must be closed even if handshake raises an unexpected error."""
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        # Handshake data that triggers a protocol error
        mock_reader.read.side_effect = [b"\x00" * 64]

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            pytest.raises(ClusterError),
        ):
            await client.find_leader()

        # The writer must have been closed to avoid socket leak
        mock_writer.close.assert_called()

    async def test_query_leader_closes_writer_on_protocol_init_error(self) -> None:
        """Writer must be closed even if DqliteProtocol construction fails."""
        from dqliteclient.exceptions import DqliteConnectionError

        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch(
                "dqliteclient.cluster.DqliteProtocol",
                side_effect=DqliteConnectionError("init failed"),
            ),
            pytest.raises(ClusterError),
        ):
            await client.find_leader()

        # Even though DqliteProtocol() failed, the writer must be closed
        mock_writer.close.assert_called()

    async def test_query_leader_does_not_hang_on_slow_wait_closed(self) -> None:
        """_query_leader must not hang if wait_closed() blocks (e.g., unresponsive peer)."""
        import asyncio

        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=0.5)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()

        # wait_closed blocks forever (simulates unresponsive peer)
        async def hang_forever():
            await asyncio.sleep(999)

        mock_writer.wait_closed = hang_forever

        from dqlitewire.messages import LeaderResponse, WelcomeResponse

        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            LeaderResponse(node_id=1, address="").encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            # This should complete within the timeout, not hang on wait_closed
            try:
                leader = await asyncio.wait_for(client.find_leader(), timeout=2.0)
                assert leader == "localhost:9001"
            except TimeoutError:
                pytest.fail(
                    "find_leader() hung because _query_leader's finally block "
                    "awaited a blocking wait_closed()"
                )

        mock_writer.close.assert_called()

    async def test_find_leader_propagates_programming_bugs(self) -> None:
        """Programming bugs (TypeError etc.) must propagate, not be swallowed
        into a generic ClusterError. ClusterError is retryable upstream, so
        stringifying a bug here amplifies it N*retries times.
        """
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        client = ClusterClient(store, timeout=0.5)

        async def buggy_query(_address: str) -> str | None:
            raise TypeError("programmer mistake")

        with (
            patch.object(client, "_query_leader", side_effect=buggy_query),
            pytest.raises(TypeError, match="programmer mistake"),
        ):
            await client.find_leader()

    async def test_find_leader_transport_error_chains_cause(self) -> None:
        """When every node yields a transport error, the final ClusterError
        must have __cause__ set so logs show the underlying reason, not just
        the generic message.
        """
        from dqliteclient.exceptions import DqliteConnectionError

        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=0.5)

        boom = DqliteConnectionError("handshake failed")

        async def failing_query(_address: str) -> str | None:
            raise boom

        with (
            patch.object(client, "_query_leader", side_effect=failing_query),
            pytest.raises(ClusterError) as exc_info,
        ):
            await client.find_leader()
        assert exc_info.value.__cause__ is boom

    async def test_find_leader_randomizes_node_order(self) -> None:
        """find_leader must not always probe the first-listed node first —
        otherwise stampedes and stale-cache biases concentrate on it.
        """
        from collections import Counter

        store = MemoryNodeStore(["n1:9001", "n2:9001", "n3:9001", "n4:9001"])
        client = ClusterClient(store, timeout=0.2)

        first_probed: list[str] = []

        async def track(address: str) -> str | None:
            first_probed.append(address)
            raise DqliteConnectionError("not leader")

        with patch.object(client, "_query_leader", side_effect=track):
            for _ in range(50):
                with contextlib.suppress(ClusterError):
                    await client.find_leader()

        # Record the first probe of each call.
        firsts = [first_probed[i] for i in range(0, len(first_probed), 4)]
        counts = Counter(firsts)
        assert len(counts) >= 2, f"find_leader always probed the same node first: {counts}"

    async def test_find_leader_probes_voters_before_non_voters(self) -> None:
        """Non-voter nodes (standby/spare) cannot become leader; probing them
        first wastes an RTT. find_leader must prefer voters.
        """
        store = MemoryNodeStore()
        # Seed with a non-voter first, then a voter.
        await store.set_nodes(
            [
                NodeInfo(node_id=2, address="spare:9002", role=2),  # spare
                NodeInfo(node_id=1, address="standby:9003", role=1),  # standby
                NodeInfo(node_id=3, address="voter1:9001", role=0),
                NodeInfo(node_id=4, address="voter2:9004", role=0),
            ]
        )
        client = ClusterClient(store, timeout=0.2)

        order: list[str] = []

        async def track(address: str) -> str | None:
            order.append(address)
            return None  # no leader known — keep probing

        from contextlib import suppress

        with (
            patch.object(client, "_query_leader", side_effect=track),
            suppress(ClusterError),
        ):
            await client.find_leader()

        # Both voters must be probed before the spare.
        voter_positions = [i for i, a in enumerate(order) if a.startswith("voter")]
        non_voter_positions = [i for i, a in enumerate(order) if not a.startswith("voter")]
        assert voter_positions, "no voters were probed"
        assert non_voter_positions, "test setup broken — no non-voters"
        assert max(voter_positions) < min(non_voter_positions), (
            f"voters should be probed first; order={order}"
        )

    async def test_connect_does_not_retry_plain_sql_errors(self) -> None:
        """OperationalError without a leader code is a SQL-level error, not
        a transport issue — connect() should NOT retry it. Otherwise a
        schema mismatch takes 5x find_leader round trips to propagate.
        """
        from dqliteclient.exceptions import OperationalError

        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=0.2)

        call_count = 0

        async def always_sql_error() -> str:
            nonlocal call_count
            call_count += 1
            raise OperationalError(1, "some sql error")

        with (
            patch.object(client, "find_leader", side_effect=always_sql_error),
            pytest.raises(OperationalError),
        ):
            await client.connect()

        assert call_count == 1, (
            f"SQL-level OperationalError must not be retried, got {call_count} attempts"
        )

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
