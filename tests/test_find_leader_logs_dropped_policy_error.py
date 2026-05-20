"""Pins for the parallel-sweep policy-rejection observability arms:

1. WARNING log emitted when one probe's policy rejection is dropped
   because another probe found a leader (commit ``b76bd14``). Without
   this log, a SIEM watching for security-adjacent signals only sees
   the per-probe DEBUG hit and misses the "the sweep won past it"
   context.

2. ``_set_last_known_leader(None)`` in the parallel-sweep all-
   policy-rejected arm (commit ``d4eb455``). Mirrors the fast-path's
   discipline: a stale cache entry that today only redirects to a
   policy-rejected target must not survive into the next call's fast
   path or it wastes one full RTT before the same raise.

3. Fast-path catch of ``ValueError`` from a third-party
   ``NodeStore`` returning a malformed address — translated to
   ``DqliteConnectionError``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_parallel_sweep_logs_dropped_policy_rejection_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One probe self-confirms as leader (no redirect); a sibling
    probe redirects to a policy-rejected target. The dropped policy
    rejection surfaces as a WARNING naming both the rejected redirect
    and the winning leader, sanitized via ``sanitize_for_log`` so a
    server-supplied address cannot split a log record."""
    addresses = ["winner:9001", "redirector:9001"]
    store = MemoryNodeStore(addresses)

    def policy(address: str) -> bool:
        # Reject the redirect target only — allow self-confirming
        # leader probes (which take the ``_addr_equiv`` short-circuit
        # in ``_check_redirect`` and never hit the policy callable).
        return "rejected" not in address

    cluster = ClusterClient(store, timeout=10.0, redirect_policy=policy)

    async def _query_leader(address: str, **_kw: object) -> str:
        if address == "winner:9001":
            return "winner:9001"  # self-confirms
        # Redirector points at a policy-rejected target.
        return "rejected-target:9001"

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    caplog.set_level(logging.WARNING, logger="dqliteclient.cluster")
    leader = await cluster.find_leader()
    assert leader == "winner:9001"

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    matching = [
        r for r in warnings if "dropped policy rejection during successful sweep" in r.message
    ]
    assert matching, (
        "Expected WARNING log naming dropped redirect + winning leader. "
        f"Got: {[r.message for r in caplog.records]}"
    )
    msg = matching[0].message
    assert "winner:9001" in msg, f"Expected winning leader address in WARN message; got: {msg!r}"


@pytest.mark.asyncio
async def test_parallel_sweep_all_policy_rejected_invalidates_cache() -> None:
    """Pin: when EVERY probe redirects to a policy-rejected target,
    the sweep raises ``ClusterPolicyError`` AND clears the leader
    cache before raising. Without the invalidation, the next call's
    fast path probes the stale cache entry, wastes one RTT, and then
    re-raises the same error. Mirror of the fast-path arm's
    ``_set_last_known_leader(None)`` discipline."""
    addresses = ["a:9001", "b:9001"]
    store = MemoryNodeStore(addresses)

    def policy(address: str) -> bool:
        return False  # reject everything

    cluster = ClusterClient(store, timeout=10.0, redirect_policy=policy)
    cluster._set_last_known_leader("stale.example.com:9001")
    assert cluster._get_last_known_leader() == "stale.example.com:9001"

    async def _query_leader(address: str, **_kw: object) -> str:
        return "redirect-target:9001"  # both nodes redirect

    cluster._query_leader = AsyncMock(side_effect=_query_leader)

    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()

    assert cluster._get_last_known_leader() is None, (
        "parallel-sweep all-policy-rejected arm must invalidate the "
        "leader cache, mirroring the fast-path discipline"
    )


def test_fast_path_catches_value_error_from_malformed_address() -> None:
    """Inspection pin: the fast-path catch tuple includes
    ``ValueError`` so a malformed address from a third-party
    NodeStore can be translated rather than escaping as bare
    ``ValueError``."""
    import inspect

    src = inspect.getsource(ClusterClient._find_leader_impl)
    assert "ValueError" in src, (
        "Fast-path catch must include ``ValueError`` to translate "
        "malformed addresses from third-party NodeStores"
    )
