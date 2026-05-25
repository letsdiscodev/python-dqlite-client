"""Pin: ``YamlNodeStore._load_from_disk`` caps and sanitises
payload-derived values in its ``ClusterError`` diagnostics.

The file-size cap bounds the overall payload at 1 MiB, but inside
that budget a single entry can be arbitrarily large. Without the
per-diagnostic cap and ``sanitize_server_text`` pass, a corrupt or
co-tenant-modified YAML store can:

* inject CR/LF/control bytes into ``str(ClusterError)`` reaching
  operator logs (mild CWE-117), and
* place multi-hundred-KB ``repr(node_id_raw)`` / ``str(e)`` into
  ``ClusterError.args[0]`` (no ``_MAX_DISPLAY_MESSAGE`` cap applies
  to the base ``DqliteError``).

This module pins the diagnostic-hardening contract.
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
    # The huge payload must be truncated rather than embedded verbatim.
    assert "truncated" in msg
    # The diagnostic should remain well under the 10 KB payload size.
    assert len(msg) < 2_000


def test_role_string_control_bytes_sanitised(tmp_path: Path) -> None:
    """CR/LF/other control bytes from a hostile payload must be
    escaped via ``sanitize_server_text`` so they cannot bleed into
    SIEM / structured-log pipelines as fake log lines.
    """
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text(
        '- {ID: 1, Address: "h:9001", Role: "foo\\rfaked"}\n',
    )

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    # Raw CR byte must NOT appear in the diagnostic.
    assert "\r" not in msg


def test_address_validate_rewrap_capped(tmp_path: Path) -> None:
    """The ``_validate_and_normalise_nodes`` rewrap at the bottom of
    the loader embeds the inner exception's ``str(e)`` — that path
    also goes through the cap+sanitise helper."""
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
    """A regular small-payload error keeps a small diagnostic — the
    cap helper is a no-op for short values."""
    yaml_file = tmp_path / "nodes.yml"
    yaml_file.write_text("- {ID: 1, Address: 'h:9001', Role: 99}\n")

    with pytest.raises(ClusterError) as exc_info:
        YamlNodeStore(yaml_file)
    msg = str(exc_info.value)
    assert "truncated" not in msg
    assert "99" in msg
