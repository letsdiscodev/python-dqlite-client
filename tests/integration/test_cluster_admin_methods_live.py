"""Integration tests for ``ClusterClient.cluster_info`` and
``transfer_leadership`` against a live 3-node cluster.

Mirrors the existing ``test_cluster_shape.py`` discipline — the unit
tests pin behaviour against mocked wire bytes, this file pins it
against the real dqlite C server's responses. A regression in either
the wire layer or the protocol layer will be caught here even if the
unit tests pass.

Runs against the python-dqlite-dev cluster (host networking,
``127.0.0.1:9001-9003`` advertised); the previous container-internal
address advertisement that blocked these tests is fixed.
"""

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
        """The fixed test cluster has three voter nodes. ``cluster_info``
        asks the leader and returns all three — id, address, role —
        in the order Raft replicates them."""
        store = MemoryNodeStore(cluster_node_addresses)
        cluster = ClusterClient(store, timeout=5.0)

        nodes = await cluster.cluster_info()

        assert len(nodes) == 3, f"expected 3 nodes, got {len(nodes)}: {nodes}"
        # All three must be voters in the test cluster.
        for node in nodes:
            assert node.role == NodeRole.VOTER, (
                f"expected voter, got {node.role} for node {node.node_id} @ {node.address}"
            )
            assert node.node_id >= 1
            assert node.address  # non-empty wire address

        ids = {node.node_id for node in nodes}
        assert len(ids) == 3, f"node ids must be unique, got {ids}"


@pytest.mark.integration
class TestTransferLeadershipAgainstLiveCluster:
    async def test_transfer_to_specific_node_changes_leader(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """End-to-end: pick a non-current voter, transfer leadership
        to it, observe the leader-id flip via ``find_leader`` +
        ``cluster_info``.

        Election convergence is asynchronous on the server; poll
        ``find_leader`` until the address points at the target node
        or the bounded timeout elapses. A clean ``transfer_leadership``
        call followed by an observable id-flip is the load-bearing
        contract this test pins.
        """
        store = MemoryNodeStore(cluster_node_addresses)
        cluster = ClusterClient(store, timeout=5.0)

        # Discover starting leader + cluster shape.
        nodes = await cluster.cluster_info()
        starting_leader_addr = await cluster.find_leader()
        starting_leader = next((n for n in nodes if n.address == starting_leader_addr), None)
        assert starting_leader is not None, (
            f"current leader address {starting_leader_addr!r} not in "
            f"cluster_info node list {nodes!r}"
        )

        # Pick a different voter as the transfer target.
        target = next(
            (n for n in nodes if n.role == NodeRole.VOTER and n.node_id != starting_leader.node_id),
            None,
        )
        assert target is not None, "no other voter available for transfer test"

        # Transfer.
        await cluster.transfer_leadership(target_node_id=target.node_id)

        # Poll for convergence — the new leader is observable when
        # ``find_leader`` returns the target's address.
        deadline = asyncio.get_event_loop().time() + 10.0
        last_seen: str | None = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                last_seen = await cluster.find_leader()
            except Exception:
                # Mid-flip the cluster may briefly have no leader; keep
                # polling within the deadline.
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

        # Restore the original leader so subsequent tests see the
        # cluster in a known state. Best-effort — if the restore
        # itself fails, the test still recorded the load-bearing
        # forward-direction contract.
        import contextlib

        with contextlib.suppress(Exception):
            await cluster.transfer_leadership(target_node_id=starting_leader.node_id)

    async def test_transfer_to_nonexistent_node_raises_operational_error(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """Server rejects a transfer to a node id not in the cluster.
        The rejection surfaces as ``OperationalError`` with the
        upstream code+message — the same translation every other
        protocol method uses."""
        store = MemoryNodeStore(cluster_node_addresses)
        cluster = ClusterClient(store, timeout=5.0)

        # Node id 999 is not in the 3-node test cluster.
        with pytest.raises(OperationalError):
            await cluster.transfer_leadership(target_node_id=999)
