"""End-to-end pins for two dqlite-specific cluster-shape contracts:

1. Connecting directly to a follower must yield ``DqliteConnectionError("...not leader...")``
   (a transport error, not SQL), so the pool's _invalidate+rotate failover path can fire.
2. ``find_leader`` must skip an unreachable first address — pinned against a real unbound
   port so a future narrowing of the ``except OSError`` clause is caught (mocks would miss it).
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
        """Direct connect to each node yields 1 success + N-1 ``DqliteConnectionError(…leader…)``.

        Retries up to 3x only when 0 successes are seen, to tolerate a brief re-election window.
        """
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
            # Mid-election: all nodes returned NOT_LEADER. Retry.
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
        """``find_leader`` tolerates a per-node failure and tries the next address. Pin the
        real ``ECONNREFUSED`` shape against port 1 (reliably unbound) — mocks would miss a
        future narrowing of the ``except`` clause."""
        store = MemoryNodeStore(["localhost:1", cluster_address])
        cluster = ClusterClient(store, timeout=5.0)
        leader = await cluster.find_leader()
        # What matters: find_leader returned non-empty rather than raising on the dead node.
        assert leader, f"find_leader returned empty leader address: {leader!r}"
