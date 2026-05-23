"""``YamlNodeStore`` malformed-config branches: OSError on read,
YAML parsing to None, role wrong type. All three are real
operator-facing failure paths; without coverage a regression
silently misbehaves (corrupt store reads as empty, weird types
crash deeper in find_leader).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


def test_oserror_on_read_raises_cluster_error(tmp_path: Path) -> None:
    """File exists but os.read raises OSError (e.g. permission
    denied mid-read). The constructor must surface the failure as
    ClusterError so the operator sees the path."""
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
    """A file whose content is the literal ``null\\n`` parses to
    Python ``None``. Treat as empty rather than rejecting — matches
    the empty-file path's tolerance."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("null\n")

    store = YamlNodeStore(yaml_file)
    # Cannot await get_nodes synchronously; check the populated
    # cache directly.
    assert store._nodes == ()


def test_role_wrong_type_raises_cluster_error(tmp_path: Path) -> None:
    """Role field that is neither int nor str (e.g. a YAML list)
    must surface as ClusterError with a clear diagnostic."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        "- {ID: 1, Address: 'h:9001', Role: [1, 2]}\n",
    )

    with pytest.raises(ClusterError, match="'Role' must be int or str"):
        YamlNodeStore(yaml_file)
