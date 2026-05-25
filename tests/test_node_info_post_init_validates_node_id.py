"""Pin: ``NodeInfo.__post_init__`` validates ``node_id`` at
construction, mirroring ``cluster._validate_node_id`` and the
YAML loader's bool/int discipline.

Pre-fix the constructor validated only ``role``. ID=0 (the upstream
"no node" sentinel), bools (int subclass), and negatives all slipped
through to ``MemoryNodeStore.set_nodes`` and on into wire-encoder
land — caught only at the membership-change call site or the server
reply, defeating the "keep the diagnostic at the construction site"
rationale used elsewhere in the codebase.
"""

from __future__ import annotations

import pytest

from dqliteclient.node_store import NodeInfo
from dqlitewire import NodeRole


def test_node_id_zero_rejected() -> None:
    with pytest.raises(ValueError, match=r"node_id must be >= 1"):
        NodeInfo(node_id=0, address="host:9000", role=NodeRole.VOTER)


def test_node_id_negative_rejected() -> None:
    with pytest.raises(ValueError, match=r"node_id must be >= 1"):
        NodeInfo(node_id=-5, address="host:9000", role=NodeRole.VOTER)


def test_node_id_bool_true_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id=True, address="host:9000", role=NodeRole.VOTER)


def test_node_id_bool_false_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id=False, address="host:9000", role=NodeRole.VOTER)


def test_node_id_float_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id=1.5, address="host:9000", role=NodeRole.VOTER)  # type: ignore[arg-type]


def test_node_id_str_rejected() -> None:
    with pytest.raises(TypeError, match=r"node_id must be int"):
        NodeInfo(node_id="1", address="host:9000", role=NodeRole.VOTER)  # type: ignore[arg-type]


def test_node_id_one_accepted() -> None:
    info = NodeInfo(node_id=1, address="host:9000", role=NodeRole.VOTER)
    assert info.node_id == 1


def test_large_node_id_accepted() -> None:
    info = NodeInfo(node_id=2**40, address="host:9000", role=NodeRole.VOTER)
    assert info.node_id == 2**40
