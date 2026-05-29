"""Client-side ``NodeInfo`` rejects unknown role values at construction, like the wire side."""

from __future__ import annotations

import pytest

from dqliteclient.node_store import NodeInfo
from dqlitewire import NodeRole


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (NodeRole.VOTER, NodeRole.VOTER),
        (NodeRole.STANDBY, NodeRole.STANDBY),
        (NodeRole.SPARE, NodeRole.SPARE),
        (0, NodeRole.VOTER),
        (1, NodeRole.STANDBY),
        (2, NodeRole.SPARE),
    ],
)
def test_node_info_accepts_canonical_roles(role: NodeRole | int, expected: NodeRole) -> None:
    node = NodeInfo(node_id=1, address="leader:9001", role=role)  # type: ignore[arg-type]
    assert node.role == expected
    assert isinstance(node.role, NodeRole)


@pytest.mark.parametrize("bogus_role", [3, 4, 999])
def test_node_info_rejects_unknown_roles(bogus_role: int) -> None:
    with pytest.raises(ValueError, match="role"):
        NodeInfo(node_id=1, address="leader:9001", role=bogus_role)  # type: ignore[arg-type]
