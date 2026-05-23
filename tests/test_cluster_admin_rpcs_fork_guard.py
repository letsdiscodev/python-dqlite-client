"""Pin: ``ClusterClient`` per-address admin paths
(``describe(address=...)``, ``set_weight(address=...)``,
``open_admin_connection``) raise ``InterfaceError`` when called
after fork — symmetric with the existing ``find_leader`` guard.

The previous discipline only guarded ``find_leader``. The
``address=<specific>`` arms bypass ``find_leader`` and the public
``open_admin_connection`` primitive opens a socket directly,
leaving three documented per-node escape hatches without the
fork-after-init contract. A child built bespoke admin tooling
against a parent-allocated instance would silently succeed for
those paths and raise the canonical ``InterfaceError`` only on
leader-routed calls — confusing for cross-driver retry middleware.

Uses ``monkeypatch.setattr`` to spoof ``_current_pid`` so the test
exercises the post-fork branch deterministically without a real
``fork()``.
"""

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
    """The guard fires BEFORE invoking the operator-supplied
    ``dial_func``, so a callable capturing parent-loop-bound state
    (e.g. ``aiohttp.ClientSession``) is never called post-fork."""
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
    """Regression pin: the existing leader-routed guard must
    survive the refactor that lifts the check into a shared
    ``_check_pid()`` helper."""
    cluster = _make_cluster()
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)
    with pytest.raises(InterfaceError, match="after fork"):
        await cluster.find_leader()
