"""Smoke test for the python-dqlite-dev testlib cluster_control fixture.

Validates the sys.path / pytest_plugins bootstrap in
``tests/integration/conftest.py`` by exercising the most basic
``TestClusterControl`` operations against the live cluster. A
failure here means either the testlib is not on sys.path (sibling
repo missing), or ``ClusterClient.cluster_info`` /
``transfer_leadership`` no longer wire through correctly.

The harder fault-injection tests in
``test_pool_concurrent_tx_leader_flip.py`` build on the same
fixture; this file is the canary.
"""

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

    # Restore the original leader so subsequent tests in this session
    # observe a deterministic starting state. Best-effort — if the
    # restore itself fails the previous-leader-was-X data still got
    # captured by this test's load-bearing assertions.
    import contextlib

    with contextlib.suppress(Exception):
        await cluster_control.transfer_leadership_to(starting_leader.node_id)
        await cluster_control.wait_for_leader_change(result.leader_after)
