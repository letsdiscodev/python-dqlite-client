"""The per-address admin paths (``describe(address=...)``, ``set_weight(address=...)``,
``open_admin_connection``) bypass ``find_leader`` and so must run their own fork guard,
raising ``InterfaceError`` after fork like the leader-routed calls do."""

from __future__ import annotations

import os

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import InterfaceError
from dqliteclient.node_store import MemoryNodeStore


def _make_cluster() -> ClusterClient:
    return ClusterClient(MemoryNodeStore(addresses=["h:9001"]), timeout=2.0)


@pytest.mark.asyncio
async def test_describe_with_address_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _make_cluster()
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await cluster.describe(address="h:9001")


@pytest.mark.asyncio
async def test_set_weight_with_address_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _make_cluster()
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await cluster.set_weight(7, address="h:9001")


@pytest.mark.asyncio
async def test_open_admin_connection_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard fires before invoking the operator-supplied ``dial_func``, so a callable
    capturing parent-loop-bound state is never called post-fork."""
    cluster = _make_cluster()
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        async with cluster.open_admin_connection("h:9001"):
            pytest.fail("should not reach")


@pytest.mark.asyncio
async def test_find_leader_still_raises_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leader-routed guard must survive lifting the check into the shared
    ``_check_pid()`` helper."""
    cluster = _make_cluster()
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await cluster.find_leader()
