"""Pin: ``_validate_and_normalise_nodes`` and ``YamlNodeStore._load_from_disk``
reject app-side / file-side inputs that exceed defensive size caps.

The wire-side ``_MAX_NODE_COUNT`` (10_000) already caps server-supplied
node lists; mirror the cap for app-supplied lists so a buggy caller
funnelling 10⁶ entries through ``set_nodes`` cannot freeze the event
loop on per-entry ``parse_address`` calls under the asyncio.Lock.

The file-side ``_MAX_YAML_NODE_STORE_BYTES`` (1 MiB) bounds eager
load of the YAML file at construction.
"""

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
    """The cap is inclusive; at exactly _MAX_NODE_COUNT entries the
    helper still accepts (matches the wire-side cap which is also
    inclusive)."""
    at_cap = [NodeInfo(i + 1, f"10.0.0.1:{9000 + i}", NodeRole.VOTER) for i in range(10_000)]
    # We don't iterate the result; just want the call not to raise.
    out = _validate_and_normalise_nodes(at_cap)
    assert len(out) == 10_000


def test_yaml_node_store_rejects_oversized_file(tmp_path: Path) -> None:
    p = tmp_path / "huge.yaml"
    # Write > 1 MiB of garbage. The size precheck fires before YAML
    # parsing so the content doesn't have to be valid YAML.
    p.write_bytes(b"x" * (2 * (1 << 20)))
    with pytest.raises(ClusterError, match="exceeds maximum size"):
        YamlNodeStore(p)
