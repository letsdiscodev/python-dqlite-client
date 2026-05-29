"""Live 3-node cluster tests for ``ClusterClient.cluster_info`` + ``transfer_leadership``,
pinning the real dqlite C server's responses (unit tests cover the mocked-wire side)."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole


@pytest.mark.integration
class TestClusterInfoAgainstLiveCluster:
    async def test_cluster_info_returns_three_voter_nodes(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """``cluster_info`` asks the leader and returns all three voter nodes."""
        store = MemoryNodeStore(cluster_node_addresses)
        cluster = ClusterClient(store, timeout=5.0)

        nodes = await cluster.cluster_info()

        assert len(nodes) == 3, f"expected 3 nodes, got {len(nodes)}: {nodes}"
        for node in nodes:
            assert node.role == NodeRole.VOTER, (
                f"expected voter, got {node.role} for node {node.node_id} @ {node.address}"
            )
            assert node.node_id >= 1
            assert node.address  # non-empty

        ids = {node.node_id for node in nodes}
        assert len(ids) == 3, f"node ids must be unique, got {ids}"


@pytest.mark.integration
class TestTransferLeadershipAgainstLiveCluster:
    async def test_transfer_to_specific_node_changes_leader(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """Transfer leadership to a non-current voter and observe the leader-id flip.

        Election convergence is async, so poll find_leader until it points at the target.
        """
        store = MemoryNodeStore(cluster_node_addresses)
        cluster = ClusterClient(store, timeout=5.0)

        nodes = await cluster.cluster_info()
        starting_leader_addr = await cluster.find_leader()
        starting_leader = next((n for n in nodes if n.address == starting_leader_addr), None)
        assert starting_leader is not None, (
            f"current leader address {starting_leader_addr!r} not in "
            f"cluster_info node list {nodes!r}"
        )

        target = next(
            (n for n in nodes if n.role == NodeRole.VOTER and n.node_id != starting_leader.node_id),
            None,
        )
        assert target is not None, "no other voter available for transfer test"

        await cluster.transfer_leadership(target_node_id=target.node_id)

        # Poll until find_leader returns the target's address.
        deadline = asyncio.get_event_loop().time() + 10.0
        last_seen: str | None = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                last_seen = await cluster.find_leader()
            except Exception:
                # Mid-flip the cluster may briefly have no leader; keep polling.
                await asyncio.sleep(0.2)
                continue
            if last_seen == target.address:
                break
            await asyncio.sleep(0.2)
        else:
            pytest.fail(
                f"leader did not converge to target {target.address!r} "
                f"within timeout; last seen: {last_seen!r}"
            )

        # Best-effort restore of the original leader for subsequent tests.
        import contextlib

        with contextlib.suppress(Exception):
            await cluster.transfer_leadership(target_node_id=starting_leader.node_id)

    async def test_transfer_to_nonexistent_node_raises_operational_error(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """Server rejects a transfer to a node id not in the cluster, surfaced as
        OperationalError."""
        store = MemoryNodeStore(cluster_node_addresses)
        cluster = ClusterClient(store, timeout=5.0)

        with pytest.raises(OperationalError):
            await cluster.transfer_leadership(target_node_id=999)
