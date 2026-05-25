"""Pin: ``YamlNodeStore._load_from_disk`` PascalCase + lowercase
fallback honours an explicit-``null`` canonical key.

``dict.get(key, default)`` only honours the default when ``key`` is
missing — not when ``key`` is present with an explicit-``None``
value. Previously an entry like ``{"ID": null, "id": 5, ...}``
resolved to ``None`` and tripped "missing 'ID'" despite the
lowercase alias providing the value. The loader now uses an
explicit ``is None`` fallback so the documented "lowercase aliases
on read" tolerance applies to the explicit-null case too.
"""

from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore
from dqlitewire import NodeRole


def test_id_pascalcase_null_falls_through_to_lowercase(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: null, id: 5, Address: 'h:9001', Role: 0}\n")

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].node_id == 5


def test_address_pascalcase_null_falls_through(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        "- {ID: 1, Address: null, address: 'h:9001', Role: 0}\n",
    )

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].address == "h:9001"


def test_role_pascalcase_null_falls_through(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        "- {ID: 1, Address: 'h:9001', Role: null, role: spare}\n",
    )

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].role == NodeRole.SPARE


def test_id_canonical_explicit_wins_over_alias(tmp_path: Path) -> None:
    """Canonical PascalCase precedence is preserved — explicit
    canonical value wins over the lowercase alias."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 5, id: 99, Address: 'h:9001', Role: 0}\n")

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].node_id == 5


def test_id_lowercase_only_still_works(tmp_path: Path) -> None:
    """Pure lowercase-alias path (canonical key not present) keeps
    working — unchanged."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {id: 5, address: 'h:9001', role: voter}\n")

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].node_id == 5


def test_id_both_missing_still_raises(tmp_path: Path) -> None:
    """Both keys missing still surfaces 'missing 'ID''."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {Address: 'h:9001', Role: voter}\n")

    with pytest.raises(ClusterError, match="missing 'ID'"):
        YamlNodeStore(yaml_file)
