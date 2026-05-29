"""Live-cluster admin-method tests (describe / set_weight / dump / membership trio).

Topology-mutating tests snapshot the starting cluster shape and restore it on exit so
subsequent tests in the session see a deterministic state.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from dqliteclient.cluster import ClusterClient, LeaderInfo, NodeMetadata
from dqliteclient.exceptions import OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole

if TYPE_CHECKING:
    from dqlitetestlib import TestClusterControl  # type: ignore[import-not-found]


def _client(addresses: list[str]) -> ClusterClient:
    store = MemoryNodeStore(addresses)
    return ClusterClient(store, timeout=5.0)


@pytest.mark.integration
class TestLeaderInfoAgainstLiveCluster:
    async def test_leader_info_returns_node_id_and_address(
        self, cluster_node_addresses: list[str]
    ) -> None:
        cluster = _client(cluster_node_addresses)
        info = await cluster.leader_info()
        assert isinstance(info, LeaderInfo)
        assert info.node_id >= 1
        assert info.address  # non-empty


@pytest.mark.integration
class TestDescribeAgainstLiveCluster:
    async def test_describe_leader_returns_node_metadata(
        self, cluster_node_addresses: list[str]
    ) -> None:
        cluster = _client(cluster_node_addresses)
        meta = await cluster.describe()
        assert isinstance(meta, NodeMetadata)
        assert meta.failure_domain >= 0
        assert meta.weight >= 0

    async def test_describe_explicit_address_targets_specific_node(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """Describe each node directly via its address; all three answer with own metadata."""
        cluster = _client(cluster_node_addresses)
        nodes = await cluster.cluster_info()
        for node in nodes:
            meta = await cluster.describe(address=node.address)
            assert isinstance(meta, NodeMetadata)


@pytest.mark.integration
class TestSetWeightAgainstLiveCluster:
    async def test_set_weight_round_trips_via_describe(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """Set the leader's weight, observe via describe, restore."""
        cluster = _client(cluster_node_addresses)
        leader_addr = await cluster.find_leader()
        original = (await cluster.describe(address=leader_addr)).weight

        try:
            await cluster.set_weight(weight=42, address=leader_addr)
            after = (await cluster.describe(address=leader_addr)).weight
            assert after == 42
        finally:
            with contextlib.suppress(Exception):
                await cluster.set_weight(weight=original, address=leader_addr)


@pytest.mark.integration
class TestDumpAgainstLiveCluster:
    async def test_dump_default_database_returns_files(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """dqlite's dump returns the default database's main file and WAL sidecar as bytes."""
        cluster = _client(cluster_node_addresses)

        # Touch the default database so it has content to dump.
        async with await cluster.connect("default") as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS dump_smoke (id INTEGER)")

        files = await cluster.dump(database="default")
        assert isinstance(files, dict)
        assert files  # non-empty
        assert any("default" in name for name in files), (
            f"expected 'default' file in dump, got {list(files)!r}"
        )
        for content in files.values():
            assert isinstance(content, bytes)

    async def test_dump_unknown_database_raises_operational_error(
        self, cluster_node_addresses: list[str]
    ) -> None:
        cluster = _client(cluster_node_addresses)
        with pytest.raises(OperationalError):
            await cluster.dump(database="this-database-does-not-exist")


@pytest.mark.integration
class TestMembershipChangesAgainstLiveCluster:
    """Live Raft membership changes against a real 4th node from the testlib's
    ``spare_node`` fixture (needed because Voter promotion requires a reachable peer to
    converge). The fixture self-restores to the 3-voter state on exit."""

    async def test_add_remove_round_trip_against_running_node(
        self,
        cluster_node_addresses: list[str],
        cluster_control: TestClusterControl,
    ) -> None:
        """End-to-end membership round-trip with a real 4th node: it self-adds as Standby via
        ``--join``, then we promote to Voter (real Raft catch-up), demote to Spare, and remove.

        add_node is driven by ``--join`` since dqlite-demo has no "wait to be added" mode;
        explicit add_node is covered by unit/protocol tests + test_add_remove_config_only.
        """
        cluster = _client(cluster_node_addresses)

        starting = await cluster.cluster_info()
        starting_ids = {n.node_id for n in starting}

        async with cluster_control.spare_node() as spare:
            assert spare.node_id not in starting_ids
            after_join = await cluster.cluster_info()
            assert any(n.node_id == spare.node_id for n in after_join), (
                f"spare {spare.node_id} not in cluster_info {after_join!r}"
            )

            # Promote to Voter — needs the running peer for Raft catch-up, else blocks forever.
            await cluster.assign_role(node_id=spare.node_id, role=NodeRole.VOTER)
            await cluster_control.wait_for_role(spare.node_id, NodeRole.VOTER, timeout=10.0)

            # Demote to Spare first; remove_node rejects removing a voter without demotion.
            await cluster.assign_role(node_id=spare.node_id, role=NodeRole.SPARE)
            await cluster_control.wait_for_role(spare.node_id, NodeRole.SPARE, timeout=10.0)

        # spare_node's __aexit__ already removed the node; pin the cluster is back to start.
        final = await cluster.cluster_info()
        assert {n.node_id for n in final} == starting_ids

    async def test_add_remove_config_only_round_trip(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """add_node + remove_node against a fake address: Raft's Add only updates the leader's
        config and never contacts the new node synchronously, so no real peer is needed."""
        cluster = _client(cluster_node_addresses)

        starting = await cluster.cluster_info()
        starting_ids = {n.node_id for n in starting}

        # An id + address outside the cluster; Raft just records the address, never contacts it.
        fake_id = max(starting_ids) + 1
        fake_addr = "127.0.0.1:9099"

        try:
            await cluster.add_node(node_id=fake_id, address=fake_addr)
            after_add = await cluster.cluster_info()
            assert any(n.node_id == fake_id for n in after_add)
        finally:
            with contextlib.suppress(Exception):
                await cluster.remove_node(node_id=fake_id)

        final = await cluster.cluster_info()
        assert {n.node_id for n in final} == starting_ids
