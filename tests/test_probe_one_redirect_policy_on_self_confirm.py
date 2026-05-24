"""Pin: ``_probe_one`` (inside ``find_leader``) applies the configured
``redirect_policy`` to EVERY confirmed-leader address, including the
self-confirm case where the probed node returns its OWN address as
the leader.

Before the fix, ``_check_redirect`` was gated behind ``if not
_addr_equiv(leader_address, node.address)``, so the policy was
silently bypassed when the cluster's current leader happened to be
the probed node. The cached fast-path correctly consults the policy
(cluster.py:867-869, 919) — confirming the maintainer intent that
redirect policy applies to ALL leader-resolution paths. The probe
self-confirm arm was the lone asymmetric path.

Canonical exposure window: the user seeds the full cluster node list
(typical) AND configures a narrower policy (regional pin / audit
mode) AND the policy-excluded node is the current leader. That
intersection IS the documented use case for an allowlist policy
("regional pin"), so the bypass is a real silent invariant
violation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.exceptions import ClusterError, ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_self_confirm_with_excluded_address_does_not_silently_return() -> None:
    """A node that self-confirms as leader but is excluded by the
    allowlist policy must NOT be returned silently. This is the
    canonical regional-pin use case.

    Setup: seed contains the excluded node only. Probing it returns
    its own address (self-confirm). Currently the policy gate is
    skipped on self-confirm and the excluded address is returned.
    After the fix, the policy gate runs and the call propagates
    ClusterPolicyError or fails the gather loop."""
    store = MemoryNodeStore(["10.0.0.99:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        # Policy excludes 10.0.0.99 — regional-pin shape (allowlist
        # mentions only 10.0.0.1, not the seed node).
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
    """Negative pin: a node that self-confirms AND is allowed by
    policy is returned unchanged. Confirms the fix doesn't over-
    reject."""
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
    """Regression: the existing redirect-different-from-probe arm
    still rejects. The fix only HOISTS the policy check out of the
    inner ``if not _addr_equiv`` — it does not remove the original
    check site."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(
        store,
        timeout=5.0,
        redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
    )

    # Probe 10.0.0.1 → "attacker.com is leader" (redirect to
    # excluded address).
    async def probe(address: str, **_: object) -> str:
        return "attacker.com:9001"

    with (
        patch.object(cc, "_query_leader", side_effect=probe),
        pytest.raises((ClusterPolicyError, ClusterError)),
    ):
        asyncio.run(cc.find_leader())


def test_no_policy_self_confirm_still_returns_leader() -> None:
    """Regression: no policy configured → self-confirm path returns
    leader address (no over-reject when there's no policy)."""
    store = MemoryNodeStore(["10.0.0.1:9001"])
    cc = ClusterClient(store, timeout=5.0)  # no redirect_policy
    with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.1:9001")):
        result = asyncio.run(cc.find_leader())
    assert result == "10.0.0.1:9001"
