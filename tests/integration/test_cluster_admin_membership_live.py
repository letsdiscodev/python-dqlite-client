"""Integration tests for the new admin methods against a live cluster.

Sister of ``test_cluster_admin_methods_live.py`` (which covers
``cluster_info`` + ``transfer_leadership``). The unit suite covers
each method's wire dispatch and validation against mocked
transports; this file pins the contract against the real dqlite C
server's responses.

Tests that mutate cluster topology (``add_node`` / ``remove_node``
/ ``assign_role``) are written to be self-restoring: they snapshot
the starting cluster shape and revert their changes on the way
out so subsequent tests in the session see a deterministic
starting state. ``set_weight`` similarly snapshots and restores.
``dump`` and ``describe`` are read-only.
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


# --- leader_info ---


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


# --- describe ---


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
        """Describe each node directly via its address; all three
        must answer with their own metadata."""
        cluster = _client(cluster_node_addresses)
        nodes = await cluster.cluster_info()
        for node in nodes:
            meta = await cluster.describe(address=node.address)
            assert isinstance(meta, NodeMetadata)


# --- set_weight ---


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


# --- dump ---


@pytest.mark.integration
class TestDumpAgainstLiveCluster:
    async def test_dump_default_database_returns_files(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """The cluster's default database has at least the main file
        and its WAL sidecar after some traffic. dqlite's dump returns
        both as bytes."""
        cluster = _client(cluster_node_addresses)

        # Touch the default database with a minimal write so it has
        # content to dump.
        async with await cluster.connect("default") as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS dump_smoke (id INTEGER)")

        files = await cluster.dump(database="default")
        assert isinstance(files, dict)
        assert files  # non-empty
        # The database file is named after the database; dqlite also
        # ships a `<name>-wal` sidecar.
        assert any("default" in name for name in files), (
            f"expected 'default' file in dump, got {list(files)!r}"
        )
        # Sanity: every value is bytes.
        for content in files.values():
            assert isinstance(content, bytes)

    async def test_dump_unknown_database_raises_operational_error(
        self, cluster_node_addresses: list[str]
    ) -> None:
        cluster = _client(cluster_node_addresses)
        with pytest.raises(OperationalError):
            await cluster.dump(database="this-database-does-not-exist")


# --- add / assign / remove (the membership trio) ---


@pytest.mark.integration
class TestMembershipChangesAgainstLiveCluster:
    """Live Raft membership changes against a real running 4th
    node spawned via the ``python-dqlite-dev`` testlib's
    ``spare_node`` primitive.

    The 4th node (``cluster/docker-compose.yml::node4``, profile
    ``spare``) self-joins via ``dqlite-demo --join`` so it lands
    in the cluster as a Standby on startup. Tests then exercise
    the wire-level lifecycle our client controls — promote to
    Voter, demote back, and remove. The Raft replication needed
    for the Voter promotion to actually converge requires a
    real, reachable peer; that is exactly what the fixture
    provides.

    Each test is self-restoring: the ``cluster_control.spare_node()``
    context manager handles ``remove_node`` + container teardown
    + data-volume wipe on exit so subsequent tests in the session
    see a deterministic 3-voter starting state.

    See ``test_add_remove_round_trip_against_running_node`` (live)
    and the ``add_node`` / ``remove_node`` /
    ``assign_role`` unit + protocol tests (mocked wire) for the
    full coverage of these methods.
    """

    async def test_add_remove_round_trip_against_running_node(
        self,
        cluster_node_addresses: list[str],
        cluster_control: TestClusterControl,
    ) -> None:
        """End-to-end membership round-trip with a real 4th node.

        The new node self-adds as Standby via ``--join``; we then
        exercise the parts of the membership API our client
        actually drives on top: ``assign_role`` (Standby → Voter
        with real Raft catch-up), ``cluster_info`` reflects the
        promotion, ``assign_role`` (Voter → Spare for a clean
        teardown), and ``remove_node`` via the context manager's
        ``__aexit__``.

        ``add_node`` is exercised by ``--join``-driven self-add
        rather than an explicit ``cluster.add_node`` call —
        ``dqlite-demo`` does not have a "wait to be added" mode.
        The unit / protocol tests cover the explicit ``add_node``
        wire dispatch, and a separate live test
        (``test_add_remove_config_only_round_trip``) exercises
        the cluster-config side of ``add_node`` + ``remove_node``
        without a real running node.
        """
        cluster = _client(cluster_node_addresses)

        starting = await cluster.cluster_info()
        starting_ids = {n.node_id for n in starting}

        async with cluster_control.spare_node() as spare:
            # Sanity: the spare auto-joined and is visible.
            assert spare.node_id not in starting_ids
            after_join = await cluster.cluster_info()
            assert any(n.node_id == spare.node_id for n in after_join), (
                f"spare {spare.node_id} not in cluster_info {after_join!r}"
            )

            # Promote to Voter — needs the running peer to catch
            # up via Raft replication. Without a real running
            # node, this would block indefinitely.
            await cluster.assign_role(node_id=spare.node_id, role=NodeRole.VOTER)
            await cluster_control.wait_for_role(spare.node_id, NodeRole.VOTER, timeout=10.0)

            # Demote back to Spare so the spare-node teardown's
            # ``remove_node`` runs without the "cannot remove
            # voter without prior demotion" error.
            await cluster.assign_role(node_id=spare.node_id, role=NodeRole.SPARE)
            await cluster_control.wait_for_role(spare.node_id, NodeRole.SPARE, timeout=10.0)

        # ``spare_node`` context manager already called
        # ``remove_node`` + tore down the container; pin that the
        # cluster is back to its starting shape.
        final = await cluster.cluster_info()
        assert {n.node_id for n in final} == starting_ids

    async def test_add_remove_config_only_round_trip(
        self, cluster_node_addresses: list[str]
    ) -> None:
        """``add_node`` + ``remove_node`` against a fake address —
        exercises the cluster-config side without a real running
        peer.

        Raft's ``Add`` only updates the leader's config; it does
        not contact the new node synchronously. So we can pin the
        wire dispatch + cluster_info reflection of an add /
        remove cycle without spawning a real 4th node. The
        complementary ``test_add_remove_round_trip_against_running_node``
        above does need a real peer because Voter promotion
        requires actual log catch-up.
        """
        cluster = _client(cluster_node_addresses)

        starting = await cluster.cluster_info()
        starting_ids = {n.node_id for n in starting}

        # Pick an id outside the starting set + an address that is
        # not a real cluster member. The address never actually
        # gets contacted as part of ``Add`` — Raft just records it.
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
