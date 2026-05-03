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

import pytest

from dqliteclient.cluster import ClusterClient, LeaderInfo, NodeMetadata
from dqliteclient.exceptions import OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole


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
    """Membership changes mutate Raft state; these tests are
    self-restoring (add a node, assign, remove) so the cluster shape
    returns to the starting (3-voter) configuration on success or
    failure.

    Skipped because a real ``add_node`` requires a running fourth
    dqlite-demo process at the announced ``host:port``; the
    python-dqlite-dev cluster has only three nodes. Pinning the
    test shape here so the moment the fixture grows a spare-node
    primitive (or the test infrastructure can spawn one), the skip
    can be lifted.

    The validation paths and protocol-level encoding are fully
    covered by the unit + protocol tests; the missing live coverage
    is "the cluster actually accepts the request and updates Raft
    membership."
    """

    @pytest.fixture(autouse=True)
    def _skip_for_now(self) -> None:
        pytest.skip(
            "Gated on a 4th-node fixture: add_node requires a real "
            "dqlite-demo listening at the announced host:port for the "
            "Raft-side join handshake. python-dqlite-dev cluster has "
            "exactly 3 nodes; lifting the skip needs either a spawnable "
            "spare-node primitive or a known unused dqlite-demo "
            "pre-running on a fourth port."
        )

    async def test_add_remove_round_trip(self, cluster_node_addresses: list[str]) -> None:
        cluster = _client(cluster_node_addresses)

        # Snapshot starting cluster.
        starting = await cluster.cluster_info()
        starting_ids = {n.node_id for n in starting}

        # Pick a free node id + address. The address would need to
        # point at a live dqlite-demo for the request to succeed —
        # see class docstring.
        new_id = max(starting_ids) + 1
        new_addr = "127.0.0.1:9099"

        try:
            await cluster.add_node(node_id=new_id, address=new_addr)
            after_add = await cluster.cluster_info()
            assert any(n.node_id == new_id for n in after_add)

            # Promote to voter, observe role change.
            await cluster.assign_role(node_id=new_id, role=NodeRole.VOTER)
            after_assign = await cluster.cluster_info()
            new_node = next(n for n in after_assign if n.node_id == new_id)
            assert new_node.role == NodeRole.VOTER
        finally:
            with contextlib.suppress(Exception):
                await cluster.remove_node(node_id=new_id)
