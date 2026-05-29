"""Pin: ``YamlNodeStore._load_from_disk`` rejects non-str ``Address`` at the loader
instead of coercing via ``str()``, which would blame a bogus literal downstream rather
than the YAML schema mismatch.
"""

from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


def test_address_int_raises_cluster_error_at_loader(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 42, Role: voter}\n")

    with pytest.raises(ClusterError, match="'Address' must be str, got int"):
        YamlNodeStore(yaml_file)


def test_address_bool_raises_cluster_error_at_loader(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: true, Role: voter}\n")

    with pytest.raises(ClusterError, match="'Address' must be str, got bool"):
        YamlNodeStore(yaml_file)


def test_address_list_raises_cluster_error_at_loader(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: [host, 9000], Role: voter}\n")

    with pytest.raises(ClusterError, match="'Address' must be str, got list"):
        YamlNodeStore(yaml_file)


def test_address_dict_raises_cluster_error_at_loader(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: {host: x, port: 9000}, Role: voter}\n")

    with pytest.raises(ClusterError, match="'Address' must be str, got dict"):
        YamlNodeStore(yaml_file)


def test_address_bracketed_ipv6_string_accepted(tmp_path: Path) -> None:
    """A quoted IPv6 ``[::1]:9000`` address loads cleanly (YAML parses it as a string)."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text('- {ID: 1, Address: "[::1]:9000", Role: voter}\n')

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].address == "[::1]:9000"


def test_address_plain_string_happy_path_unchanged(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text('- {ID: 1, Address: "127.0.0.1:9001", Role: voter}\n')

    store = YamlNodeStore(yaml_file)
    assert store._nodes[0].address == "127.0.0.1:9001"
    assert store._nodes[0].node_id == 1
