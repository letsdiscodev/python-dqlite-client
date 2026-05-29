"""A malformed redirect address (parse_address ValueError) from one peer
must not sabotage the parallel ``find_leader`` sweep: the narrow except
tuples must catch ValueError so healthy siblings still win."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_malformed_redirect_target_does_not_sabotage_sweep() -> None:
    store = MemoryNodeStore(["node-a:9001", "node-b:9002"])
    client = ClusterClient(store, timeout=0.5)

    async def _query(address: str, **_kw: Any) -> str | None:
        if "node-a" in address:
            return "bad@host:9001"  # malformed redirect target
        if "node-b" in address:
            return "node-b:9002"  # healthy self-confirming leader
        # Verifying the malformed hint hits parse_address in production;
        # simulate its ValueError directly.
        raise ValueError(f"malformed address: {address!r}")

    with patch.object(client, "_query_leader", side_effect=_query):
        leader = await client.find_leader()

    assert leader == "node-b:9002"


@pytest.mark.asyncio
async def test_malformed_redirect_does_not_leak_valueerror_to_connect() -> None:
    """When EVERY peer returns a malformed redirect, ``connect()``
    must surface a wrapped ``ClusterError`` / ``DqliteConnectionError``,
    not bare ``ValueError``."""
    from dqliteclient.exceptions import ClusterError, DqliteConnectionError

    store = MemoryNodeStore(["node-a:9001"])
    client = ClusterClient(store, timeout=0.3)

    async def _query(address: str, **_kw: Any) -> str | None:
        if "node-a" in address:
            return "bad@host:9001"
        raise ValueError(f"malformed address: {address!r}")

    with (
        patch.object(client, "_query_leader", side_effect=_query),
        pytest.raises((ClusterError, DqliteConnectionError)),
    ):
        await client.connect()
