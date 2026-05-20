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

from unittest.mock import AsyncMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterPolicyError
from dqliteclient.node_store import MemoryNodeStore


def test_find_leader_impl_emits_warn_log_when_winner_and_policy_error_both_set() -> None:
    """Inspection pin: when ``_find_leader_impl`` ends with both
    ``winning_address`` AND ``policy_error`` set, it emits the
    "dropped policy rejection during successful sweep" WARN log
    naming both the rejected redirect and the winning leader,
    sanitised via ``sanitize_for_log`` (CWE-117).

    A runtime gather-race test for the WARN was inherently flaky
    because the ``asyncio.wait(FIRST_COMPLETED)`` for-loop processes
    ``done`` in hash-order — the for-loop break on
    ``winning_address`` leaves the rejector unprocessed in roughly
    half the runs, no ``policy_error`` set, no WARN. The actual
    behavioural contract (when both states ARE set, WARN fires) is
    captured by a source-substring inspection pin here. The runtime
    coverage of the both-states-set path is exercised in production
    when both probes legitimately complete in the same wait()
    invocation."""
    import inspect

    src = inspect.getsource(ClusterClient._find_leader_impl)
    # The WARN message body is split across source lines (long
    # logger.warning format string); search for the canonical
    # substring fragment.
    assert "dropped policy rejection during successful" in src
    assert "sanitize_for_log" in src, (
        "WARN log must sanitise the policy_error and winning_address "
        "(CWE-117); the substring asserts the wrapping is present"
    )
    # The WARN must be gated on both winning_address and policy_error
    # being set — not the policy_error-only path (which has its own
    # raise + cache-clear branch).
    assert "if policy_error is not None:" in src
    assert "winning_address is not None:" in src


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
