"""Pin: ``_load_from_disk`` caps and sanitises payload-derived values in its
``ClusterError`` diagnostics. Without it a corrupt entry can inject CR/LF into logs
(CWE-117) or place hundreds of KB into ``ClusterError.args[0]`` (uncapped on DqliteError).
"""

from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


def test_role_string_oversize_payload_truncated(tmp_path: Path) -> None:
    yaml_file = tmp_path / "nodes.yml"
    huge = "A" * 10_000
    yaml_file.write_text(
        "- {ID: 1, Address: 'h:9001', Role: '" + huge + "'}\n",
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" in msg
    assert len(msg) < 2_000


def test_role_string_control_bytes_sanitised(tmp_path: Path) -> None:
    """Control bytes must be escaped so they can't forge fake log lines."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        '- {ID: 1, Address: "h:9001", Role: "foo\\rfaked"}\n',
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "\r" not in msg


def test_address_validate_rewrap_capped(tmp_path: Path) -> None:
    """The ``_validate_and_normalise_nodes`` rewrap also caps+sanitises the inner str(e)."""
    yaml_file = tmp_path / "nodes.yml"
    huge_addr = "x" * 10_000 + ":9000"
    yaml_file.write_text(
        '- {ID: 1, Address: "' + huge_addr + '", Role: voter}\n',
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" in msg
    assert len(msg) < 2_000


def test_happy_path_diagnostic_unchanged(tmp_path: Path) -> None:
    """The cap helper is a no-op for short values."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: 99}\n")

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" not in msg
    assert "99" in msg
