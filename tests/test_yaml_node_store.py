"""``YamlNodeStore`` is byte-compatible with go-dqlite (PascalCase keys, integer Role).

Reads also accept lowercase keys and string Role aliases; writes emit canonical format.
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


@pytest.mark.asyncio
async def test_round_trip_canonical_format(tmp_path: Path) -> None:
    """Regression vector: format drift makes the Go reader reject."""
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
    """Load go-dqlite's exact format; load-bearing for the cross-language interop claim."""
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


def test_missing_file_returns_empty_tuple(tmp_path: Path) -> None:
    """go-dqlite parity: tolerate a missing file at construction time."""
    path = tmp_path / "does-not-exist.yaml"
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes == ()

    asyncio.run(run())
    assert not path.exists()  # no eager file creation


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
    """Surface operator-facing errors at construction, not deep inside find_leader."""
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
    """``bool`` slips through ``isinstance(_, int)``; reject so ``Role: true`` != StandBy."""
    path = tmp_path / "bad-role-bool.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n  Role: true\n", encoding="utf-8")
    with pytest.raises(ClusterError, match="bool"):
        YamlNodeStore(path)


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
    """Omitting ``Role:`` defaults to Voter, matching MemoryNodeStore's seed-list path."""
    path = tmp_path / "no-role.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n", encoding="utf-8")
    store = YamlNodeStore(path)

    async def run() -> None:
        nodes = await store.get_nodes()
        assert nodes[0].role == NodeRole.VOTER

    asyncio.run(run())


@pytest.mark.asyncio
async def test_set_nodes_file_mode_0600(tmp_path: Path) -> None:
    """Mode 0600 (go-dqlite parity): addresses can leak private DSNs / internal hosts."""
    path = tmp_path / "perms.yaml"
    store = YamlNodeStore(path)
    await store.set_nodes([NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER)])

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected mode 0o600, got {oct(mode)}"


@pytest.mark.asyncio
async def test_set_nodes_atomic_rename_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` fails, the original file stays intact and the temp file is cleaned up."""
    path = tmp_path / "atomic.yaml"
    store = YamlNodeStore(path)
    await store.set_nodes([NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER)])
    original_text = path.read_text(encoding="utf-8")

    def raising_replace(*_args: object, **_kw: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("dqliteclient.node_store.os.replace", raising_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        await store.set_nodes([NodeInfo(node_id=2, address="node2:9002", role=NodeRole.VOTER)])

    assert path.read_text(encoding="utf-8") == original_text
    siblings = [p for p in tmp_path.iterdir() if p != path]
    assert siblings == [], f"unexpected temp files left behind: {siblings}"


@pytest.mark.asyncio
async def test_set_nodes_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent dir must be fsync'd after rename, else the directory entry can revert
    on power loss even though the new content is durable."""
    import os as _os
    import stat as _stat

    path = tmp_path / "durable.yaml"
    store = YamlNodeStore(path)

    fsync_targets: list[int] = []
    real_fsync = _os.fsync

    def recording_fsync(fd: int) -> None:
        fsync_targets.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("dqliteclient.node_store.os.fsync", recording_fsync)

    await store.set_nodes([NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER)])

    # Two fsyncs expected: temp file + parent dir (fds can't be introspected post-close).
    assert len(fsync_targets) >= 2, (
        f"expected at least 2 fsync calls (temp file + parent dir), got {len(fsync_targets)}"
    )

    assert _stat.S_ISDIR(_os.stat(tmp_path).st_mode)


@pytest.mark.asyncio
async def test_concurrent_set_nodes_last_writer_wins(tmp_path: Path) -> None:
    """Concurrent ``set_nodes`` calls serialise via the asyncio.Lock; no torn mix."""
    path = tmp_path / "concurrent.yaml"
    store = YamlNodeStore(path)

    payload_a = [NodeInfo(node_id=1, address="a:9001", role=NodeRole.VOTER)]
    payload_b = [NodeInfo(node_id=2, address="b:9001", role=NodeRole.VOTER)]

    await asyncio.gather(store.set_nodes(payload_a), store.set_nodes(payload_b))

    final = await store.get_nodes()
    assert final == tuple(payload_a) or final == tuple(payload_b)


def test_pyyaml_missing_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deferred import inside __init__ raises ImportError pointing at the extra,
    leaving the rest of the package importable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="yaml-store"):
        YamlNodeStore("/tmp/does-not-matter.yaml")


@pytest.mark.asyncio
async def test_cluster_client_can_use_yaml_node_store(tmp_path: Path) -> None:
    """End-to-end: ``ClusterClient`` works with a ``YamlNodeStore`` unchanged."""
    from dqliteclient import ClusterClient

    path = tmp_path / "cluster.yaml"
    path.write_text("- ID: 1\n  Address: node1:9001\n  Role: 0\n", encoding="utf-8")
    store = YamlNodeStore(path)
    cluster = ClusterClient(store, timeout=1.0)
    nodes = await cluster._node_store.get_nodes()
    assert nodes[0].address == "node1:9001"


def test_yaml_node_store_exported_from_top_level() -> None:
    from dqliteclient import YamlNodeStore as TopLevelYNS

    assert TopLevelYNS is YamlNodeStore
