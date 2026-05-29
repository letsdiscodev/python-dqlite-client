"""Node-count and YAML-file-size caps reject oversized app-side and file-side inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import (
    NodeInfo,
    YamlNodeStore,
    _validate_and_normalise_nodes,
)
from dqlitewire import NodeRole


def test_validate_rejects_too_many_nodes() -> None:
    big = [NodeInfo(i + 1, f"10.0.0.1:{9000 + i}", NodeRole.VOTER) for i in range(10_001)]
    with pytest.raises(ValueError, match="too many nodes"):
        _validate_and_normalise_nodes(big)


def test_validate_accepts_at_cap() -> None:
    """The cap is inclusive: exactly _MAX_NODE_COUNT entries is accepted."""
    at_cap = [NodeInfo(i + 1, f"10.0.0.1:{9000 + i}", NodeRole.VOTER) for i in range(10_000)]
    out = _validate_and_normalise_nodes(at_cap)
    assert len(out) == 10_000


def test_yaml_node_store_rejects_oversized_file(tmp_path: Path) -> None:
    p = tmp_path / "huge.yaml"
    # Size precheck fires before YAML parsing, so the content need not be valid YAML.
    p.write_bytes(b"x" * (2 * (1 << 20)))
    with pytest.raises(ClusterError, match="exceeds maximum size"):
        YamlNodeStore(p)
