"""Smoke/canary test for the testlib ``cluster_control`` fixture: validates the sys.path /
pytest_plugins bootstrap by exercising basic ``TestClusterControl`` ops on the live cluster.
A failure means the testlib is not on sys.path or cluster_info/transfer_leadership broke."""

from __future__ import annotations

import pytest


@pytest.mark.integration
async def test_cluster_control_observes_three_voter_nodes(cluster_control: object) -> None:
    """The fixed test cluster has three voter nodes; ``cluster_info``
    via the fixture should report all three."""
    from dqlitetestlib import TestClusterControl  # type: ignore[import-not-found]

    from dqlitewire import NodeRole

    assert isinstance(cluster_control, TestClusterControl)
    nodes = await cluster_control.cluster_info()
    assert len(nodes) == 3, f"expected 3 nodes, got {nodes!r}"
    for n in nodes:
        assert n.role == NodeRole.VOTER
        assert n.node_id >= 1
        assert n.address  # non-empty


@pytest.mark.integration
async def test_cluster_control_force_leader_flip_converges(cluster_control: object) -> None:
    """End-to-end: force a leader flip, observe convergence to the
    target. Restores the cluster's original leader on the way out so
    subsequent tests see a stable starting state."""
    from dqlitetestlib import TestClusterControl

    assert isinstance(cluster_control, TestClusterControl)

    starting_leader = await cluster_control.current_leader_node()

    result = await cluster_control.force_leader_flip()
    assert result.leader_after == result.target.address
    assert result.target.node_id != starting_leader.node_id

    # Best-effort restore of the original leader for subsequent tests.
    import contextlib

    with contextlib.suppress(Exception):
        await cluster_control.transfer_leadership_to(starting_leader.node_id)
        await cluster_control.wait_for_leader_change(result.leader_after)
