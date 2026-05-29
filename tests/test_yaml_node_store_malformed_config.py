"""``YamlNodeStore`` malformed-config branches: OSError on read, YAML None, role wrong type."""

from pathlib import Path
from unittest.mock import patch

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


def test_oserror_on_read_raises_cluster_error(tmp_path: Path) -> None:
    """os.read raising OSError must surface as ClusterError so the operator sees the path."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("[]")

    def fake_read(_fd: int, _n: int) -> bytes:
        raise PermissionError("permission denied")

    with (
        patch("dqliteclient.node_store.os.read", fake_read),
        pytest.raises(ClusterError, match="cannot read"),
    ):
        YamlNodeStore(yaml_file)


def test_null_yaml_content_returns_empty(tmp_path: Path) -> None:
    """Literal ``null`` parses to ``None``; treat as empty, matching the empty-file path."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("null\n")

    store = YamlNodeStore(yaml_file)
    assert store._nodes == ()


def test_role_wrong_type_raises_cluster_error(tmp_path: Path) -> None:
    """A Role that is neither int nor str (e.g. a YAML list) must surface as ClusterError."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        "- {ID: 1, Address: 'h:9001', Role: [1, 2]}\n",
    )

    with pytest.raises(ClusterError, match="'Role' must be int or str"):
        YamlNodeStore(yaml_file)
