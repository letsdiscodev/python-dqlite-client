"""Pin: ``_find_leader_impl`` sorts nodes by role strictly ascending
(VOTER → STANDBY → SPARE), not into a binary
voter / non-voter bucket. Standbys are probed before spares because
they participate in heartbeats and are more likely to know the
current leader.

Matches go-dqlite's ``connector.go::connectAttempt`` discipline.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore, NodeInfo
from dqlitewire import NodeRole


def _build_cluster_with_mixed_roles() -> ClusterClient:
    addresses = [
        "v1:9001",  # VOTER
        "sb1:9001",
        "sb2:9001",
        "sb3:9001",  # 3x STANDBY
        "sp1:9001",
        "sp2:9001",
        "sp3:9001",  # 3x SPARE
    ]
    store = MemoryNodeStore(addresses)
    # Re-write the role assignments — MemoryNodeStore's default is
    # VOTER for everyone; we want a mixed-role distribution.
    role_map = {
        "v1:9001": NodeRole.VOTER,
        "sb1:9001": NodeRole.STANDBY,
        "sb2:9001": NodeRole.STANDBY,
        "sb3:9001": NodeRole.STANDBY,
        "sp1:9001": NodeRole.SPARE,
        "sp2:9001": NodeRole.SPARE,
        "sp3:9001": NodeRole.SPARE,
    }
    # NodeInfo is frozen; rebuild the internal tuple.
    rebuilt = tuple(
        NodeInfo(node_id=i + 1, address=addr, role=role_map[addr])
        for i, addr in enumerate(addresses)
    )
    store._nodes = rebuilt
    return ClusterClient(store, concurrent_leader_conns=1, attempt_timeout=2.0)


@pytest.mark.asyncio
async def test_find_leader_probes_standby_before_spare() -> None:
    """With ``concurrent_leader_conns=1`` the probe order is fully
    sequential, making the sort discipline observable. All STANDBY
    entries must precede any SPARE entry."""
    cluster = _build_cluster_with_mixed_roles()

    probe_order: list[str] = []

    async def _record(addr: str, **_kw: object) -> str | None:
        probe_order.append(addr)
        # Every node returns its own address (self-confirms) so the
        # first probe wins.
        return None  # no leader known — drive the full sweep

    with (
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_record)),
        contextlib.suppress(Exception),  # all-no-leader → ClusterError; we want probe order
    ):
        await cluster.find_leader()

    sb_positions = [i for i, a in enumerate(probe_order) if a.startswith("sb")]
    sp_positions = [i for i, a in enumerate(probe_order) if a.startswith("sp")]
    v_positions = [i for i, a in enumerate(probe_order) if a.startswith("v")]

    # Voters first.
    if v_positions and sb_positions:
        assert max(v_positions) < min(sb_positions), (
            f"VOTER must precede STANDBY: probe order = {probe_order}"
        )
    # Then standbys.
    if sb_positions and sp_positions:
        assert max(sb_positions) < min(sp_positions), (
            f"All STANDBY must precede any SPARE: probe order = {probe_order}"
        )
