"""Leader-redirect allowlist policy (ISSUE-04).

A compromised peer can return any address as the leader; the client
used to open a TCP connection to whatever it was told. ClusterClient
now takes an optional ``redirect_policy`` callable that authorizes
each redirect target; a convenience ``allowlist_policy`` helper builds
one from a static list of addresses.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


class TestRedirectPolicy:
    def test_accepts_seeded_redirect(self) -> None:
        store = MemoryNodeStore(["10.0.0.1:9001", "10.0.0.2:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(["10.0.0.1:9001", "10.0.0.2:9001"]),
        )
        # Simulate: probing 10.0.0.1 reports 10.0.0.2 as leader.
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.2:9001")):
            result = asyncio.run(cc.find_leader())
        assert result == "10.0.0.2:9001"

    def test_rejects_unknown_redirect(self) -> None:
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
        )
        # A compromised peer redirects to an attacker-controlled host.
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="attacker.com:9001")):
            with pytest.raises(ClusterError, match="rejected"):
                asyncio.run(cc.find_leader())

    def test_no_policy_means_any_redirect_accepted(self) -> None:
        """Default (None) policy preserves legacy behavior."""
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(store, timeout=5.0)
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="anywhere.invalid:9001")):
            result = asyncio.run(cc.find_leader())
        assert result == "anywhere.invalid:9001"

    def test_self_leader_bypasses_policy(self) -> None:
        """If the queried node is the leader (returns its own address),
        the redirect policy doesn't apply — the address is already in the
        seed list by definition."""
        store = MemoryNodeStore(["10.0.0.1:9001"])
        # Policy that rejects everything — but the node returning its own
        # address isn't a real redirect, so it's accepted.
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=lambda _a: False,
        )
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.1:9001")):
            result = asyncio.run(cc.find_leader())
        assert result == "10.0.0.1:9001"
