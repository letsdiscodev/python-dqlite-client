"""Pin: ``YamlNodeStore._load_from_disk`` rejects float ``ID`` values.

Like ``int(True)``, ``int(3.7) == 3`` silently coerces; the guard diverts floats to a
ClusterError so a float ID can't truncate to a different node id (or the ``0`` sentinel).
"""

from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


def test_id_float_truncating_rejected(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 3.7, Address: 'h:9001', Role: voter}\n")

    with pytest.raises(ClusterError, match="'ID' must be integer"):
        YamlNodeStore(yaml_file)


def test_id_float_zero_rejected(tmp_path: Path) -> None:
    """``int(0.5) == 0`` would land on the 'no node' sentinel."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 0.5, Address: 'h:9001', Role: voter}\n")

    with pytest.raises(ClusterError, match="'ID' must be integer"):
        YamlNodeStore(yaml_file)


def test_id_integer_valued_float_still_rejected(tmp_path: Path) -> None:
    """Rejection keys on the YAML *type* being float, not whether the value is integral."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1.0, Address: 'h:9001', Role: voter}\n")

    with pytest.raises(ClusterError, match="'ID' must be integer"):
        YamlNodeStore(yaml_file)


def test_id_plain_integer_happy_path(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: voter}\n")

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].node_id == 1


def test_id_string_int_still_accepted(tmp_path: Path) -> None:
    """Preserve the ``int("5")`` path accepting hand-edited string-typed integer IDs."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: '5', Address: 'h:9001', Role: voter}\n")

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].node_id == 5


def test_id_bool_still_rejected(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: true, Address: 'h:9001', Role: voter}\n")

    with pytest.raises(ClusterError, match="'ID' must be integer"):
        YamlNodeStore(yaml_file)
