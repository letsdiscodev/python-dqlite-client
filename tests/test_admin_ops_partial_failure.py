"""Pin: ``ClusterClient.add_node`` partial-failure recovery path.

``add_node`` is a two-phase op (ADD then optional ASSIGN). The
docstring at ``cluster.py:add_node`` documents the recovery shape:
when ADD succeeds but the follow-up ASSIGN raises, the cluster's
Raft log records the new node as ``SPARE`` and the exception
propagates. The caller's recovery is to re-run ``assign_role``
against the same node_id to converge to the requested role.

Pre-this-pin no standalone test exercised the partial-failure path.
This file pins the behaviour the docstring promises so future
refactors of the two-phase shape don't silently regress.

Scope (per reviewer KEEP-WITH-NOTES on the umbrella issue): ONLY
the ``add_node`` ADD-landed-ASSIGN-raised case. ``remove_node`` /
``set_weight`` / ``transfer_leadership`` partial-failure pins are
sibling work; one-issue-one-commit per project rules.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import OperationalError
from dqliteclient.node_store import MemoryNodeStore
from dqlitewire import NodeRole

_FakeOpenConnection = Callable[[str, int], Awaitable[tuple[MagicMock, MagicMock]]]


def _make_cluster() -> ClusterClient:
    store = MemoryNodeStore(["localhost:9001"])
    return ClusterClient(store, timeout=0.5)


def _patch_admin_connection() -> tuple[_FakeOpenConnection, MagicMock]:
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    async def fake_open_connection(
        host: str, port: int, **_kwargs: object
    ) -> tuple[MagicMock, MagicMock]:
        return reader, writer

    return fake_open_connection, writer


@pytest.mark.asyncio
async def test_add_node_assign_failure_propagates_and_invalidates_leader_cache() -> None:
    """ADD lands, ASSIGN raises → the original assign error propagates
    to the caller AND the leader cache is invalidated by the finally
    so a follow-up recovery call (``assign_role``) re-discovers the
    leader instead of trusting a possibly-stale cache.
    """
    cluster = _make_cluster()

    # Pre-seed the leader cache so we can pin its invalidation.
    cluster._set_last_known_leader("node1:9001")
    assert cluster._get_last_known_leader() == "node1:9001"

    fake_proto = MagicMock()
    fake_proto.handshake = AsyncMock()
    fake_proto.negotiate_protocol_only = AsyncMock()
    fake_proto.add = AsyncMock()  # ADD lands
    fake_proto.assign = AsyncMock(
        side_effect=OperationalError("transient leader-flip on assign", 1)
    )
    fake_open, _ = _patch_admin_connection()

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto),
        pytest.raises(OperationalError, match="transient leader-flip on assign"),
    ):
        await cluster.add_node(node_id=42, address="node42:9001", role=NodeRole.VOTER)

    # ADD ran exactly once; ASSIGN ran exactly once and was the
    # raiser. The node is now in the cluster's Raft membership at
    # the implicit-SPARE landing role.
    fake_proto.add.assert_awaited_once_with(42, "node42:9001")
    fake_proto.assign.assert_awaited_once_with(42, NodeRole.VOTER)

    # add_node's finally invalidates the leader cache on both
    # success and failure (a leader-flip-induced ASSIGN failure may
    # have left the cache pointing at the rejecter).
    assert cluster._get_last_known_leader() is None


@pytest.mark.asyncio
async def test_add_node_partial_failure_recovery_via_assign_role_converges() -> None:
    """Pin the docstring-promised recovery: after ADD-then-ASSIGN
    partial failure, the caller's ``assign_role(node_id, role)`` call
    runs the missing second phase and lands the desired role.

    Two-phase scenario:
    1. ``add_node(role=VOTER)``: ADD lands, ASSIGN raises → caller
       sees the exception.
    2. Caller catches and runs ``assign_role(node_id, NodeRole.VOTER)``
       to converge — this time both leader-discovery and the ASSIGN
       round-trip succeed.

    Pin: the recovery ``assign_role`` round-trip dispatches the same
    ``(node_id, role)`` against the leader and resolves cleanly.
    """
    cluster = _make_cluster()

    # Phase 1: add_node with ADD-success / ASSIGN-failure.
    fake_proto_phase1 = MagicMock()
    fake_proto_phase1.handshake = AsyncMock()
    fake_proto_phase1.negotiate_protocol_only = AsyncMock()
    fake_proto_phase1.add = AsyncMock()
    fake_proto_phase1.assign = AsyncMock(side_effect=OperationalError("assign failed", 1))
    fake_open_phase1, _ = _patch_admin_connection()

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open_phase1),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto_phase1),
        pytest.raises(OperationalError, match="assign failed"),
    ):
        await cluster.add_node(node_id=99, address="node99:9001", role=NodeRole.VOTER)

    # ADD landed, ASSIGN raised — the cluster has node 99 at SPARE.
    fake_proto_phase1.add.assert_awaited_once_with(99, "node99:9001")
    fake_proto_phase1.assign.assert_awaited_once_with(99, NodeRole.VOTER)

    # Phase 2: caller runs assign_role to converge. This is the
    # documented recovery path.
    fake_proto_phase2 = MagicMock()
    fake_proto_phase2.handshake = AsyncMock()
    fake_proto_phase2.negotiate_protocol_only = AsyncMock()
    fake_proto_phase2.assign = AsyncMock()  # this time, succeeds
    fake_open_phase2, _ = _patch_admin_connection()

    with (
        patch.object(cluster, "find_leader", AsyncMock(return_value="node1:9001")),
        patch("dqliteclient._dial.open_connection_with_keepalive", new=fake_open_phase2),
        patch("dqliteclient.cluster.DqliteProtocol", return_value=fake_proto_phase2),
    ):
        await cluster.assign_role(node_id=99, role=NodeRole.VOTER)

    # The recovery ASSIGN dispatched with the original (node_id, role).
    fake_proto_phase2.assign.assert_awaited_once_with(99, NodeRole.VOTER)
