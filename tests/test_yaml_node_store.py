"""``YamlNodeStore`` is byte-compatible with go-dqlite's
``NewYamlNodeStore`` (PascalCase keys, integer Role enum). Mixed-
language deployments (Go service + Python service sharing one
bootstrap file) require this exact shape.

The on-read parser also accepts lowercase keys and string Role
aliases for ergonomics with hand-edited files. Writes always emit
the canonical go-dqlite format.
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

import pytest
import yaml

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import NodeInfo, YamlNodeStore
from dqlitewire import NodeRole

# ---------------------------------------------------------------- canonical format


@pytest.mark.asyncio
async def test_round_trip_canonical_format(tmp_path: Path) -> None:
    """Write 3 NodeInfo, re-read raw bytes, assert PascalCase keys
    + integer Role values. Pre-fix the format was lowercase strings;
    the regression vector is "format drift makes Go reader reject"."""
    path = tmp_path / "cluster.yaml"
    store = YamlNodeStore(path)
    await store.set_nodes(
        [
            NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="node2:9002", role=NodeRole.STANDBY),
            NodeInfo(node_id=3, address="node3:9003", role=NodeRole.SPARE),
        ]
    )

    raw = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)

    assert parsed == [
        {"ID": 1, "Address": "node1:9001", "Role": 0},
        {"ID": 2, "Address": "node2:9002", "Role": 1},
        {"ID": 3, "Address": "node3:9003", "Role": 2},
    ]


@pytest.mark.asyncio
async def test_go_dqlite_format_fixture_loads(tmp_path: Path) -> None:
    """Hand-craft a byte string in go-dqlite's exact format and load
    it. Load-bearing for the cross-language interop claim — pre-fix
    the lowercase-keys parser silently accepted ``id``/``address``/
    ``role`` and would have rejected this fixture."""
    path = tmp_path / "go-format.yaml"
    path.write_text(
        "- ID: 1\n"
        "  Address: node1:9001\n"
        "  Role: 0\n"
        "- ID: 2\n"
        "  Address: node2:9002\n"
        "  Role: 1\n"
        "- ID: 3\n"
        "  Address: node3:9003\n"
        "  Role: 2\n",
        encoding="utf-8",
    )

    store = YamlNodeStore(path)
    nodes = await store.get_nodes()

    assert len(nodes) == 3
    assert nodes[0] == NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER)
    assert nodes[1] == NodeInfo(node_id=2, address="node2:9002", role=NodeRole.STANDBY)
    assert nodes[2] == NodeInfo(node_id=3, address="node3:9003", role=NodeRole.SPARE)


# ---------------------------------------------------------------- empty / missing


def test_missing_file_returns_empty_tuple(tmp_path: Path) -> None:
    """Matches go-dqlite's ``NewYamlNodeStore`` which tolerates a
    missing file at construction time."""
    path = tmp_path / "does-not-exist.yaml"
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes == ()

    asyncio.run(run())
    assert not path.exists()  # eager file creation NOT performed


def test_empty_file_returns_empty_tuple(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes == ()

    asyncio.run(run())


def test_whitespace_only_file_returns_empty_tuple(tmp_path: Path) -> None:
    path = tmp_path / "whitespace.yaml"
    path.write_text("   \n\t\n   ", encoding="utf-8")
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes == ()

    asyncio.run(run())


# ---------------------------------------------------------------- malformed input


def test_corrupt_yaml_raises_clusterror(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.yaml"
    path.write_text("- ID: 1\n  Address: [unclosed", encoding="utf-8")
    with pytest.raises(ClusterError, match="malformed YAML"):
        YamlNodeStore(path)


def test_top_level_not_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "scalar.yaml"
    path.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="top-level must be a YAML list"):
        YamlNodeStore(path)


def test_entry_not_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong-shape.yaml"
    path.write_text("- hello\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="must be a mapping"):
        YamlNodeStore(path)


def test_missing_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "no-id.yaml"
    path.write_text("- Address: node1:9001\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="missing 'ID'"):
        YamlNodeStore(path)


def test_missing_address_raises(tmp_path: Path) -> None:
    path = tmp_path / "no-address.yaml"
    path.write_text("- ID: 1\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="missing 'Address'"):
        YamlNodeStore(path)


def test_invalid_address_rejected_at_load(tmp_path: Path) -> None:
    """Same syntactic validation as ``MemoryNodeStore`` — surface
    operator-facing errors at construction, not deep inside
    ``find_leader``."""
    path = tmp_path / "bad-addr.yaml"
    path.write_text("- ID: 1\n  Address: not-an-address\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="invalid"):
        YamlNodeStore(path)


def test_id_must_be_integer_or_int_string(tmp_path: Path) -> None:
    """``int(...)`` accepts numeric strings but not arbitrary text."""
    path = tmp_path / "bad-id.yaml"
    path.write_text("- ID: abc\n  Address: node1:9001\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="'ID' must be integer"):
        YamlNodeStore(path)


def test_role_invalid_integer_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-role-int.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n  Role: 99\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="not a valid NodeRole"):
        YamlNodeStore(path)


def test_role_invalid_string_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-role-str.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n  Role: superleader\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="not one of voter/stand-by/spare"):
        YamlNodeStore(path)


def test_role_bool_rejected(tmp_path: Path) -> None:
    """``bool`` slips through ``isinstance(_, int)``; explicit reject
    so ``Role: true`` doesn't silently become ``StandBy``."""
    path = tmp_path / "bad-role-bool.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n  Role: true\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="bool"):
        YamlNodeStore(path)


# ---------------------------------------------------------------- ergonomic accept


def test_lowercase_keys_accepted(tmp_path: Path) -> None:
    """Hand-edited files with lowercase keys must still load."""
    path = tmp_path / "lowercase.yaml"
    path.write_text("- id: 1\n  address: node1:9001\n  role: 0\n", encoding="utf-8")
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes == (NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER),)

    asyncio.run(run())


@pytest.mark.parametrize(
    "value,expected",
    [
        ("voter", NodeRole.VOTER),
        ("Voter", NodeRole.VOTER),
        ("VOTER", NodeRole.VOTER),
        ("stand-by", NodeRole.STANDBY),
        ("standby", NodeRole.STANDBY),
        ("stand_by", NodeRole.STANDBY),
        ("StandBy", NodeRole.STANDBY),
        ("spare", NodeRole.SPARE),
        ("Spare", NodeRole.SPARE),
    ],
)
def test_role_string_aliases_accepted(tmp_path: Path, value: str, expected: NodeRole) -> None:
    path = tmp_path / "string-role.yaml"
    path.write_text(f"- ID: 1\n  Address: node1:9001\n  Role: {value}\n", encoding="utf-8")
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes[0].role == expected

    asyncio.run(run())


def test_role_missing_defaults_to_voter(tmp_path: Path) -> None:
    """Matches ``MemoryNodeStore``'s seed-list path which assumes
    Voter. Operator omits ``Role:`` and gets the most-common
    default."""
    path = tmp_path / "no-role.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n", encoding="utf-8")
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes[0].role == NodeRole.VOTER

    asyncio.run(run())


# ---------------------------------------------------------------- atomicity / file mode


@pytest.mark.asyncio
async def test_set_nodes_file_mode_0600(tmp_path: Path) -> None:
    """Match go-dqlite's ``renameio.WriteFile(..., 0600)``. Bootstrap
    files often contain implicit secrets via the addresses they
    reference (private DSNs, internal hostnames)."""
    path = tmp_path / "perms.yaml"
    store = YamlNodeStore(path)
    await store.set_nodes([NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER)])

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected mode 0o600, got {oct(mode)}"


@pytest.mark.asyncio
async def test_set_nodes_atomic_rename_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` fails, the original file must remain
    intact and the temp file must be cleaned up."""
    path = tmp_path / "atomic.yaml"
    store = YamlNodeStore(path)
    # Initial successful write to populate the original file.
    await store.set_nodes([NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER)])
    original_text = path.read_text(encoding="utf-8")

    def raising_replace(*_args: object, **_kw: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("dqliteclient.node_store.os.replace", raising_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        await store.set_nodes([NodeInfo(node_id=2, address="node2:9002", role=NodeRole.VOTER)])

    # Original file unchanged.
    assert path.read_text(encoding="utf-8") == original_text
    # Temp file cleaned up — only the original exists.
    siblings = [p for p in tmp_path.iterdir() if p != path]
    assert siblings == [], f"unexpected temp files left behind: {siblings}"


@pytest.mark.asyncio
async def test_concurrent_set_nodes_last_writer_wins(tmp_path: Path) -> None:
    """Two concurrent ``set_nodes`` calls must serialise via the
    asyncio.Lock; the final state is one of the two payloads, not
    a torn mix."""
    path = tmp_path / "concurrent.yaml"
    store = YamlNodeStore(path)

    payload_a = [NodeInfo(node_id=1, address="a:9001", role=NodeRole.VOTER)]
    payload_b = [NodeInfo(node_id=2, address="b:9001", role=NodeRole.VOTER)]

    await asyncio.gather(store.set_nodes(payload_a), store.set_nodes(payload_b))

    final = await store.get_nodes()
    assert final == tuple(payload_a) or final == tuple(payload_b)


# ---------------------------------------------------------------- pyyaml import


def test_pyyaml_missing_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pyyaml installed, constructing ``YamlNodeStore``
    must raise ``ImportError`` directing the operator to the
    extra. The deferred import happens inside ``__init__`` so the
    rest of the package keeps importing cleanly."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="yaml-store"):
        YamlNodeStore("/tmp/does-not-matter.yaml")


# ---------------------------------------------------------------- integration via cluster


@pytest.mark.asyncio
async def test_cluster_client_can_use_yaml_node_store(tmp_path: Path) -> None:
    """End-to-end: ``ClusterClient`` works with a ``YamlNodeStore``
    just like a ``MemoryNodeStore``. The Protocol-based ``NodeStore``
    interface means no other code change is needed."""
    from dqliteclient import ClusterClient

    path = tmp_path / "cluster.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n  Role: 0\n", encoding="utf-8")
    store = YamlNodeStore(path)
    cluster = ClusterClient(store, timeout=1.0)
    nodes = await cluster._node_store.get_nodes()
    assert nodes[0].address == "node1:9001"


# ---------------------------------------------------------------- top-level export


def test_yaml_node_store_exported_from_top_level() -> None:
    """``YamlNodeStore`` is importable from the package root."""
    from dqliteclient import YamlNodeStore as TopLevelYNS

    assert TopLevelYNS is YamlNodeStore
