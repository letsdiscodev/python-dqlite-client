"""``ClusterClient.open_admin_connection`` wraps dial-side
``OSError`` / ``TimeoutError`` as ``DqliteConnectionError`` so admin
RPC callers see the documented exception class. Mirrors
``DqliteConnection.connect``'s discipline.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_open_admin_connection_wraps_oserror_as_dqlite_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]), timeout=2.0)

    async def refusing_open(*_args: object, **_kwargs: object):
        raise ConnectionRefusedError("simulated ECONNREFUSED")

    monkeypatch.setattr("dqliteclient._dial.open_connection_with_keepalive", refusing_open)

    with pytest.raises(DqliteConnectionError, match="Failed to connect"):
        async with cluster.open_admin_connection("localhost:9001"):
            pass


@pytest.mark.asyncio
async def test_open_admin_connection_wraps_timeout_as_dqlite_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = ClusterClient(MemoryNodeStore(["localhost:9001"]), timeout=0.1)

    async def slow_open(*_args: object, **_kwargs: object):
        await asyncio.sleep(2.0)
        raise AssertionError("should have timed out")

    monkeypatch.setattr("dqliteclient._dial.open_connection_with_keepalive", slow_open)

    # The dial stalls under ``slow_open``; with ``dial_timeout`` and
    # ``attempt_timeout`` both defaulted from ``timeout=0.1`` they fire
    # at nearly the same instant. Either the inner ``dial_timeout`` arm
    # surfaces (``Connection to ... timed out``) or the outer
    # ``attempt_timeout`` arm surfaces (``Admin handshake to ...
    # exceeded attempt_timeout``). The contract is "a
    # ``DqliteConnectionError`` mentioning the stall" — both messages
    # satisfy it.
    with pytest.raises(DqliteConnectionError, match=r"timed out|attempt_timeout"):
        async with cluster.open_admin_connection("localhost:9001"):
            pass
