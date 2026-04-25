"""Integration tests for dqlite-specific cluster-shape behaviors.

dqlite is a Raft-replicated distributed SQLite. Two cluster-shape
contracts are unique to dqlite (no analog in postgres / mysql / single-
node sqlite client libraries) and have unit-test coverage with mocked
wire responses but no end-to-end pin against the real cluster:

1. **Connect-to-follower returns a transport-level error.**
   Connecting directly (no leader-discovery) to any of the cluster
   nodes yields exactly one success (the leader) and a
   ``DqliteConnectionError("...no longer leader...")`` from each
   follower. The follower rejects ``OPEN_DATABASE`` with the
   ``SQLITE_IOERR_NOT_LEADER`` code, which the client translates to
   ``DqliteConnectionError`` so the pool's ``_invalidate``+rotate
   path can fire (a SQL-level error wouldn't trigger failover).
   Unit test ``test_connection.py:854`` mocks the wire byte stream;
   this pins the contract against the real C dqlite server's reply.

2. **``find_leader`` skips an unreachable first address.**
   The probe loop tolerates an immediate ``ECONNREFUSED`` from one
   node and falls through to the remaining addresses. Unit tests
   patch ``asyncio.open_connection`` to raise ``OSError``; this pins
   the kernel's actual ``ConnectionRefusedError`` shape against an
   unbound TCP port — catching a future narrowing of the ``except
   OSError`` clause that would pass unit tests but break in
   production.

The ``cluster_node_addresses`` fixture comes from
``conftest.py`` and is configurable via ``DQLITE_TEST_CLUSTER_NODES``.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import connect
from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.integration
class TestConnectToFollowerReturnsLeaderError:
    async def test_connect_to_each_node_exactly_one_is_leader(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """The 3-node Raft cluster has exactly one leader at any time;
        direct connect (bypassing ``ClusterClient.find_leader``) to
        each node yields 1 success + N-1 ``DqliteConnectionError("…leader…")``.

        Tolerates the brief no-leader window during a Raft re-election
        by retrying the probe up to 3 times (only when 0 successes are
        observed — a (1, N-1) split is the steady-state shape and any
        other split is a real bug)."""
        successes: list[str] = []
        leader_errors: list[tuple[str, str]] = []
        for _attempt in range(3):
            successes = []
            leader_errors = []
            for addr in cluster_node_addresses:
                conn = None
                try:
                    conn = await connect(addr, timeout=5.0)
                    successes.append(addr)
                except DqliteConnectionError as e:
                    msg = str(e).lower()
                    assert "leader" in msg, (
                        f"connect to {addr} raised DqliteConnectionError "
                        f"without 'leader' in the message: {e!r}"
                    )
                    leader_errors.append((addr, str(e)))
                finally:
                    if conn is not None:
                        await conn.close()
            if len(successes) == 1:
                break
            # Mid-election window: all nodes returned NOT_LEADER. Retry.
            await asyncio.sleep(0.5)

        assert len(successes) == 1, (
            f"expected exactly 1 leader, got successes={successes} "
            f"errors={leader_errors} (after retry)"
        )
        assert len(leader_errors) == len(cluster_node_addresses) - 1, (
            f"expected {len(cluster_node_addresses) - 1} follower errors, got {leader_errors}"
        )


@pytest.mark.integration
class TestFindLeaderSkipsUnreachableNode:
    async def test_find_leader_falls_through_unreachable_first_address(
        self, cluster_address: str
    ) -> None:
        """``ClusterClient.find_leader`` probes addresses in order
        (after a stable shuffle) and tolerates a per-node failure to
        try the next one. Pin the real-network ``ECONNREFUSED`` shape
        against an unbound port — the unit tests use mocked OSError,
        which would not catch a future narrowing of the ``except``
        clause to ``BrokenPipeError``-only or similar.

        Port 1 is the IANA-reserved tcpmux port and is reliably
        unbound on a developer workstation."""
        store = MemoryNodeStore(["localhost:1", cluster_address])
        cluster = ClusterClient(store, timeout=5.0)
        leader = await cluster.find_leader()
        # The leader address may be the container-internal
        # ``0.0.0.0:9001`` (separate fixture-bug tracked elsewhere);
        # what matters here is that ``find_leader`` returned a
        # non-empty address rather than raising on the dead first
        # node.
        assert leader, f"find_leader returned empty leader address: {leader!r}"
