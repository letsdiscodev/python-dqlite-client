"""``_probe_one`` applies ``redirect_policy`` to EVERY confirmed-leader address,
including self-confirm. Previously ``_check_redirect`` was gated behind
``if not _addr_equiv(...)``, silently bypassing the policy when the probed node
was itself the leader — breaking the allowlist ("regional pin") use case."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.exceptions import ClusterError, ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_self_confirm_with_excluded_address_does_not_silently_return() -> None:
    """A self-confirming node excluded by the allowlist must not be returned."""
    store = MemoryNodeStore(["10.0.0.99:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        # Policy excludes the seed node 10.0.0.99 — regional-pin shape.
        redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
    )

    async def probe(address: str, **_: object) -> str:
        return "10.0.0.99:9001"

    with (
        patch.object(cc, "_query_leader", side_effect=probe),
        pytest.raises((ClusterPolicyError, ClusterError)),
    ):
        asyncio.run(cc.find_leader())


def test_self_confirm_with_allowed_address_returns_leader() -> None:
    """A self-confirming node allowed by policy is returned unchanged."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
    )

    with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.1:9001")):
        result = asyncio.run(cc.find_leader())
    assert result == "10.0.0.1:9001"


def test_redirect_to_excluded_still_rejected_regression() -> None:
    """Regression: the redirect-different-from-probe arm still rejects."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
    )

    # Probe 10.0.0.1 redirects to the excluded attacker.com.
    async def probe(address: str, **_: object) -> str:
        return "attacker.com:9001"

    with (
        patch.object(cc, "_query_leader", side_effect=probe),
        pytest.raises((ClusterPolicyError, ClusterError)),
    ):
        asyncio.run(cc.find_leader())


def test_no_policy_self_confirm_still_returns_leader() -> None:
    """Regression: no policy configured, self-confirm returns the leader."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)
    with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.1:9001")):
        result = asyncio.run(cc.find_leader())
    assert result == "10.0.0.1:9001"
