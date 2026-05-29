"""The node stores raise InterfaceError on get_nodes/set_nodes after fork.

Without the guard a child inherits the parent-loop-bound asyncio.Lock in a held state
(deadlock) or reads a stale snapshot. Tests spoof getpid to reach the branch without fork().
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dqliteclient.exceptions import InterfaceError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_memory_store_get_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryNodeStore(addresses=["h:9001"])
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.get_nodes()


@pytest.mark.asyncio
async def test_memory_store_set_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryNodeStore(addresses=["h:9001"])
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.set_nodes([NodeInfo(1, "h:9001", NodeRole.VOTER)])


@pytest.mark.asyncio
async def test_yaml_store_get_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cluster.yaml"
    path.write_text("- Address: h:9001\n  ID: 1\n  Role: voter\n")
    store = YamlNodeStore(path)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.get_nodes()


@pytest.mark.asyncio
async def test_yaml_store_set_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cluster.yaml"
    path.write_text("- Address: h:9001\n  ID: 1\n  Role: voter\n")
    store = YamlNodeStore(path)
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.set_nodes([NodeInfo(1, "h:9002", NodeRole.VOTER)])
