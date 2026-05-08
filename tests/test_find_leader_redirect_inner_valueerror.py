"""Pin: a malformed redirect address returned by one cluster node must not
sabotage the parallel ``find_leader`` sweep.

Threat model: a hostile or buggy peer responds to ``LeaderRequest`` with
a ``LeaderResponse(node_id=N, address=<garbage>)`` where ``<garbage>``
is a string ``parse_address`` rejects (e.g. embedded ``@`` or NUL).
``_query_leader`` returns the raw string; ``_probe_one`` then calls
``_verify_redirect`` which calls back into ``_query_leader``, which
calls ``open_connection(...)`` → ``parse_address(...)`` →
``ValueError``. The narrow ``except`` tuples in ``_verify_redirect``
and ``_probe_one`` did not include ``ValueError``, so the error
escaped the per-node probe, was classified as ``unexpected_exc`` in
the gather loop, cancelled healthy sibling probes, and re-raised —
defeating ``find_leader`` for every caller until the hostile node was
removed from the store. ``ClusterClient.connect()``'s outer ``except``
also did not include ``ValueError``, so the bare exception leaked past
the PEP 249 wrap.

Sibling precedent: the constructor path was already addressed (a
``DqliteConnection(leader, ...)`` constructor raising ``ValueError``
on a malformed address is wrapped to ``ClusterPolicyError``). This
pin covers the inner-sweep counterpart.
"""

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
            # node-a's response advertises a malformed redirect target.
            return "bad@host:9001"
        if "node-b" in address:
            # node-b is a healthy self-confirming leader.
            return "node-b:9002"
        # ``_verify_redirect`` invokes ``_query_leader`` against the
        # malformed hint — that path runs through ``open_connection``
        # in production, which calls ``parse_address`` and raises
        # ``ValueError``. Simulate it directly here.
        raise ValueError(f"malformed address: {address!r}")

    with patch.object(client, "_query_leader", side_effect=_query):
        # Should NOT raise ValueError; the sibling probe (node-b) wins.
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
