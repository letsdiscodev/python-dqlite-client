"""Tests for cluster management."""

import contextlib
from typing import Any
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

        # Upstream raft_leader sets id and address atomically: a voter
        # that IS the leader returns its own id AND its own address
        # (never (nonzero, "")).
        responses = [
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            LeaderResponse(node_id=1, address="localhost:9001").encode(),
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

    async def test_find_leader_logs_per_node_attempts_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """find_leader's retry loop should emit a DEBUG log for each
        node's failure. Without per-node tracing, a slow leader-discovery
        stall is only visible as a single accumulated ClusterError
        string; operators cannot distinguish one slow node from a
        cluster-wide problem."""
        import logging

        from dqliteclient.exceptions import DqliteConnectionError

        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        client = ClusterClient(store, timeout=1.0)

        async def always_fail(address: str, **kw: Any) -> str | None:
            raise DqliteConnectionError(f"probe of {address} refused")

        caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
        with (
            patch.object(client, "_query_leader", side_effect=always_fail),
            pytest.raises(ClusterError),
        ):
            await client.find_leader()

        messages = [rec.message for rec in caplog.records if rec.levelno == logging.DEBUG]
        assert any("localhost:9001" in m for m in messages), (
            f"DEBUG log must mention each attempted node, got: {messages}"
        )
        assert any("localhost:9002" in m for m in messages), (
            f"DEBUG log must mention each attempted node, got: {messages}"
        )

    async def test_query_leader_awaits_wait_closed_on_success(self) -> None:
        """The leader-probe socket should close cleanly: ``close()`` is
        fire-and-forget, so a bare ``writer.close()`` leaves the
        transport in FIN-WAIT until the OS reclaims it. Awaiting
        ``wait_closed()`` with a bounded timeout matches the asyncio
        documented full-close idiom and keeps FD pressure low under
        heavy connect churn (pool warm-up after a leader flip)."""
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
            LeaderResponse(node_id=1, address="localhost:9001").encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            await client.find_leader()

        mock_writer.close.assert_called()
        mock_writer.wait_closed.assert_awaited()

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

    async def test_find_leader_skips_node_with_bad_handshake(self) -> None:
        """When a node's handshake round-trip fails (malformed response
        triggers a wire-level ProtocolError), ``find_leader`` must move
        on to the next node and succeed. Guards the skip-and-continue
        loop at cluster.py:109-132 against regressions — a bug in the
        except tuple would turn one bad peer into a cluster-wide failure.
        """
        store = MemoryNodeStore(["localhost:9001", "localhost:9002"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        from dqlitewire.messages import LeaderResponse, WelcomeResponse

        # Node A responds with 64 bytes of zeros: that decodes as a
        # FAILURE-typed frame with a zero-byte body, and FailureResponse.
        # decode_body raises DecodeError because the body is too short
        # for the uint64 code. The wire ProtocolError is wrapped into
        # the client ProtocolError by DqliteProtocol._read_response and
        # caught by find_leader, which moves on to Node B.
        #
        # Node B responds with a valid Welcome + Leader pair, so
        # find_leader returns Node B's address.
        responses = [
            b"\x00" * 64,
            WelcomeResponse(heartbeat_timeout=15000).encode(),
            LeaderResponse(node_id=2, address="localhost:9002").encode(),
        ]
        mock_reader.read.side_effect = responses

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            leader = await client.find_leader()

        assert leader in {"localhost:9001", "localhost:9002"}
        # Two open_connection attempts expected (one per node); the
        # failing node's writer was closed before the skip.
        assert mock_writer.close.call_count >= 2

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
            LeaderResponse(node_id=1, address="localhost:9001").encode(),
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

        async def buggy_query(_address: str, **_kwargs: object) -> str | None:
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

        async def failing_query(_address: str, **_kwargs: object) -> str | None:
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

        async def track(address: str, **_kwargs: object) -> str | None:
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

        async def track(address: str, **_kwargs: object) -> str | None:
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

        async def always_sql_error(**_kwargs: object) -> str:
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


class TestConnectMaxAttempts:
    """connect() exposes a max_attempts parameter.

    The previous hardcoded ``max_attempts=3`` forced operators to patch
    the library to tune retry behavior. The default is unchanged; the
    knob simply becomes adjustable.
    """

    async def test_max_attempts_defaults_to_three(self) -> None:
        from dqliteclient.cluster import _DEFAULT_CONNECT_MAX_ATTEMPTS

        assert _DEFAULT_CONNECT_MAX_ATTEMPTS == 3

    async def test_max_attempts_override_honored(self) -> None:
        store = MemoryNodeStore(["localhost:1"])  # unreachable
        client = ClusterClient(store, timeout=0.1)

        call_count = [0]

        async def fake_find_leader(**_kwargs: object) -> str:
            call_count[0] += 1
            raise DqliteConnectionError("unreachable")

        client.find_leader = fake_find_leader  # type: ignore[method-assign]

        with contextlib.suppress(DqliteConnectionError):
            await client.connect(max_attempts=5)
        assert call_count[0] == 5, f"Expected 5 attempts with max_attempts=5, got {call_count[0]}"

    async def test_max_attempts_zero_rejected(self) -> None:
        store = MemoryNodeStore(["localhost:1"])
        client = ClusterClient(store, timeout=0.1)
        with pytest.raises(ValueError, match=">= 1"):
            await client.connect(max_attempts=0)


class TestConnectObservability:
    """Per-attempt failures are logged at DEBUG for diagnosis."""

    async def test_failed_attempts_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        store = MemoryNodeStore(["localhost:1"])  # unreachable
        client = ClusterClient(store, timeout=0.1)

        async def fake_find_leader(**_kwargs: object) -> str:
            raise DqliteConnectionError("simulated")

        client.find_leader = fake_find_leader  # type: ignore[method-assign]

        caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")
        with contextlib.suppress(DqliteConnectionError):
            await client.connect(max_attempts=2)

        # Every attempt should emit a debug log.
        attempt_logs = [r for r in caplog.records if "connect attempt" in r.message]
        assert len(attempt_logs) == 2, (
            f"Expected 2 per-attempt log lines, got {len(attempt_logs)}: "
            f"{[r.message for r in attempt_logs]}"
        )


class TestQueryLeaderTrustsHeartbeat:
    """_query_leader forwards the trust_server_heartbeat flag."""

    async def test_flag_propagates_to_probe_protocol(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        captured: dict[str, object] = {}

        class FakeProto:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured.update(kwargs)

            async def handshake(self) -> None:
                pass

            async def get_leader(self) -> tuple[int, str]:
                return (1, "localhost:9001")

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
        ):
            await client._query_leader("localhost:9001", trust_server_heartbeat=True)

        assert captured.get("trust_server_heartbeat") is True

    async def test_flag_default_false(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        captured: dict[str, object] = {}

        class FakeProto:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured.update(kwargs)

            async def handshake(self) -> None:
                pass

            async def get_leader(self) -> tuple[int, str]:
                return (1, "localhost:9001")

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
        ):
            await client._query_leader("localhost:9001")

        assert captured.get("trust_server_heartbeat") is False


class TestQueryLeaderRejectsUnreachableCombo:
    """A (nonzero id, empty address) response is a protocol violation;
    reject it instead of silently substituting the queried address.
    """

    async def test_nonzero_id_with_empty_address_raises_protocol_error(self) -> None:
        from dqliteclient.exceptions import ProtocolError

        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        class FakeProto:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def handshake(self) -> None:
                pass

            async def get_leader(self) -> tuple[int, str]:
                return (42, "")

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
            pytest.raises(ProtocolError),
        ):
            await client._query_leader("localhost:9001")

    async def test_find_leader_multi_node_all_no_leader_known(self) -> None:
        """Parity with the single-node case in
        ``test_find_leader_no_leader_known``: when every node in a
        multi-node store returns ``None`` from the leader probe, the
        retry loop must exhaust and raise ``ClusterError`` rather than
        hang or return ``None``. Pin the exhaustion path so a future
        refactor of the loop cannot regress into a silent-succeed.
        """
        store = MemoryNodeStore(["node-a:9001", "node-b:9002", "node-c:9003"])
        client = ClusterClient(store, timeout=0.2)

        call_count = 0

        async def no_leader(address: str, **_kw: Any) -> str | None:
            nonlocal call_count
            call_count += 1
            return None

        with (
            patch.object(client, "_query_leader", side_effect=no_leader),
            pytest.raises(ClusterError, match="Could not find leader"),
        ):
            await client.find_leader()

        # Every node must have been probed — if the loop short-circuited
        # on the first ``None`` it would only hit one.
        assert call_count == 3

    async def test_find_leader_mixed_no_leader_and_failure(self) -> None:
        """Mix a no-leader-known response with a transport failure. Both
        error paths are exhausted before the final ClusterError raises.
        """
        store = MemoryNodeStore(["node-a:9001", "node-b:9002"])
        client = ClusterClient(store, timeout=0.2)

        async def mixed(address: str, **_kw: Any) -> str | None:
            if "node-a" in address:
                return None  # no-leader-known
            raise DqliteConnectionError("probe refused")

        with (
            patch.object(client, "_query_leader", side_effect=mixed),
            pytest.raises(ClusterError, match="Could not find leader"),
        ):
            await client.find_leader()

    async def test_malformed_redirect_is_debug_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Parity with ISSUE-219: the point-of-detection DEBUG breadcrumb
        must include the queried address AND the bad (node_id, address)
        pair so an operator can correlate the discovery stall in logs
        alone, not only via the surfaced ProtocolError.
        """
        import logging

        from dqliteclient.exceptions import ProtocolError

        store = MemoryNodeStore(["node-3:9000"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        class FakeProto:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def handshake(self) -> None:
                pass

            async def get_leader(self) -> tuple[int, str]:
                return (5, "")

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
            caplog.at_level(logging.DEBUG, logger="dqliteclient.cluster"),
            pytest.raises(ProtocolError),
        ):
            await client._query_leader("node-3:9000")

        matching = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and "malformed redirect" in r.getMessage()
            and "node-3:9000" in r.getMessage()
        ]
        assert matching, f"expected DEBUG 'malformed redirect' with address; got {caplog.records!r}"

    async def test_zero_id_empty_address_returns_none(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=1.0)

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        class FakeProto:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def handshake(self) -> None:
                pass

            async def get_leader(self) -> tuple[int, str]:
                return (0, "")

        with (
            patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)),
            patch("dqliteclient.cluster.DqliteProtocol", FakeProto),
        ):
            result = await client._query_leader("localhost:9001")

        assert result is None


class TestClusterErrorMessageTruncation:
    """Per-node error snippets are capped before concatenation so a
    verbose server message cannot inflate the ClusterError payload to
    O(N * M).
    """

    async def test_large_per_node_error_is_truncated(self) -> None:
        store = MemoryNodeStore(["a:9001", "b:9001", "c:9001"])
        client = ClusterClient(store, timeout=1.0)

        huge = "x" * 50_000

        async def fake_query(_address: str, **_kwargs: object) -> str | None:
            raise DqliteConnectionError(huge)

        with (
            patch.object(client, "_query_leader", side_effect=fake_query),
            pytest.raises(ClusterError) as exc_info,
        ):
            await client.find_leader()

        message = str(exc_info.value)
        # Each per-node snippet is capped to ~200 chars + truncation
        # marker; total upper bound well under the raw 150k.
        assert len(message) < 3_000
        assert "truncated" in message


class TestTryConnectNarrowExcept:
    """Programming bugs (TypeError, AttributeError, …) in the connect
    path must propagate without being muted by the DEBUG-log instrument.
    """

    async def test_type_error_propagates(self) -> None:
        store = MemoryNodeStore(["localhost:9001"])
        client = ClusterClient(store, timeout=0.1)

        async def fake_find_leader(**_kwargs: object) -> str:
            raise TypeError("programming bug")

        client.find_leader = fake_find_leader  # type: ignore[method-assign]

        with pytest.raises(TypeError, match="programming bug"):
            await client.connect(max_attempts=1)
