"""Pin: ``MemoryNodeStore`` and ``YamlNodeStore`` raise
``InterfaceError`` on ``get_nodes`` / ``set_nodes`` after fork.

The node-store classes hold a parent-loop-bound ``asyncio.Lock``
and (for ``YamlNodeStore``) an in-memory tuple loaded from disk in
the parent process. Without the fork-after-init guard, the
dominant failure modes are:

* Multi-threaded parent forks with one thread inside
  ``set_nodes`` (the dbapi sync layer's daemon loop thread is the
  obvious path) — the child inherits the lock in a
  permanently-held state and the first child ``acquire()`` blocks
  forever. The dbapi resolve-leader cache lock has an
  ``os.register_at_fork(after_in_child=...)`` hook for the same
  failure mode; the node-store locks were the missing twins.
* Silent stale-snapshot reads from a child running on a fresh
  loop where the parent's data is no longer authoritative.

Uses ``monkeypatch.setattr`` to spoof ``_current_pid`` so the
post-fork branch is reachable deterministically without a real
``fork()``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dqliteclient import connection as _conn_mod
from dqliteclient.exceptions import InterfaceError
from dqliteclient.node_store import MemoryNodeStore, NodeInfo, YamlNodeStore
from dqlitewire import NodeRole


@pytest.mark.asyncio
async def test_memory_store_get_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryNodeStore(addresses=["h:9001"])
    monkeypatch.setattr(_conn_mod, "_current_pid", os.getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.get_nodes()


@pytest.mark.asyncio
async def test_memory_store_set_nodes_after_fork_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryNodeStore(addresses=["h:9001"])
    monkeypatch.setattr(_conn_mod, "_current_pid", os.getpid() + 1)
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
    monkeypatch.setattr(_conn_mod, "_current_pid", os.getpid() + 1)
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
    monkeypatch.setattr(_conn_mod, "_current_pid", os.getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await store.set_nodes([NodeInfo(1, "h:9002", NodeRole.VOTER)])
