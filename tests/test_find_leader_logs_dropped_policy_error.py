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

import asyncio
import logging
from typing import Any
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


@pytest.mark.asyncio
async def test_dropped_policy_warn_fires_when_both_states_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Behavioural pin: when one probe redirects to a policy-rejected
    target and ANOTHER probe self-confirms as leader, the WARN log
    must fire and name both the rejected redirect and the winning
    leader, routed through ``sanitize_for_log`` (CWE-117).

    The WARN fires only when both probes resolve in the same
    ``asyncio.wait(FIRST_COMPLETED)`` iteration AND the rejector is
    iterated before the winner (the winner's ``_LeaderHit`` arm
    ``break``s the for-loop). The original runtime test was flaky
    because ``done`` is an unordered set — iteration order depends
    on task ``id()`` hashes.

    Deterministic approach: monkey-patch ``asyncio.wait`` to return
    ``done`` as a list-ordered set (rejector first) so the gather
    loop always processes the policy rejection before the leader
    hit. This pins the WARN's wiring (sanitize_for_log routing,
    both-address interpolation) without depending on hash order.
    """
    real_wait = asyncio.wait

    class _OrderedDone(list[asyncio.Task[object]]):
        """List subclass that ``_find_leader_impl``'s
        ``for task in done`` iterates in insertion order. The
        production code only iterates ``done`` and does not branch
        on its set-ness, so substituting a list keeps the contract.
        """

    async def _wait_for_all_then_order(
        fs: Any, *args: Any, **kwargs: Any
    ) -> tuple[_OrderedDone, set[Any]]:
        # Override ``return_when=FIRST_COMPLETED`` with
        # ``ALL_COMPLETED`` so both probes resolve before we return.
        # The for-loop then sees both tasks in the same iteration,
        # with the rejector first via insertion-order on
        # ``_OrderedDone``. This is the deterministic equivalent of
        # the race the original substring-pin'd test was trying to
        # catch.
        new_kwargs = dict(kwargs)
        new_kwargs["return_when"] = asyncio.ALL_COMPLETED
        done_orig, pending_orig = await real_wait(fs, *args, **new_kwargs)
        ordered = _OrderedDone()
        rest: list[Any] = []
        for t in done_orig:
            if not t.cancelled() and t.exception() is not None:
                ordered.append(t)
            else:
                rest.append(t)
        ordered.extend(rest)
        return ordered, pending_orig

    async def _query_leader(address: str, **_kw: object) -> str | None:
        if address == "rejector:9001":
            return "rejected-target:9001"
        return "winner:9001"

    async def _verify_redirect(address: str, **_kw: object) -> str | None:
        return address

    def _policy(address: str) -> bool:
        return "rejected" not in address

    store = MemoryNodeStore(["rejector:9001", "winner:9001"])
    cluster = ClusterClient(store, timeout=10.0, redirect_policy=_policy)
    cluster._query_leader = AsyncMock(side_effect=_query_leader)
    cluster._verify_redirect = AsyncMock(side_effect=_verify_redirect)

    caplog.set_level(logging.WARNING, logger="dqliteclient.cluster")
    # ``asyncio.wait`` is module-level; patch the binding inside the
    # cluster module so _find_leader_impl picks up the wrapper.
    from unittest.mock import patch

    with patch("dqliteclient.cluster.asyncio.wait", _wait_for_all_then_order):
        leader = await asyncio.wait_for(cluster.find_leader(), timeout=2.0)
    assert leader == "winner:9001"

    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "dropped policy rejection during successful" in r.getMessage()
    ]
    assert matching, (
        "WARN must fire when both winning_address and policy_error are "
        "observed in the same find_leader sweep"
    )
    # The message must name BOTH the rejected redirect target and the
    # winning leader so a SIEM watching for security-adjacent signals
    # captures the full context.
    rendered = matching[0].getMessage()
    assert "rejected-target:9001" in rendered, (
        f"WARN must name the rejected redirect target: got {rendered!r}"
    )
    assert "winner:9001" in rendered, f"WARN must name the winning leader address: got {rendered!r}"


@pytest.mark.asyncio
async def test_dropped_policy_warn_sanitises_crlf_tainted_address(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Load-bearing CWE-117 pin: the WARN's address interpolation
    must route the policy-error text through ``sanitize_for_log`` so
    a CRLF-tainted redirect target cannot split the log record.

    The rejector's ``ClusterPolicyError`` carries the verbatim
    server-supplied address (via ``raw_message=``); the WARN
    formatter wraps ``str(policy_error)`` in ``sanitize_for_log``,
    which escapes ``\\n`` / ``\\r`` / ``\\t``. The behavioural
    assertion: the rendered WARN message does NOT contain a raw
    newline, and the escaped form is present.
    """
    # Tainted address with embedded CRLF — a hostile peer could
    # smuggle this via a LeaderResponse redirect target.
    tainted = "evil.example.com:9001\r\nINJECTED log line"

    real_wait = asyncio.wait

    class _OrderedDone(list[asyncio.Task[object]]):
        """List subclass for the gather loop's
        ``for task in done`` iteration — see the sibling test."""

    async def _wait_for_all_then_order(
        fs: Any, *args: Any, **kwargs: Any
    ) -> tuple[_OrderedDone, set[Any]]:
        new_kwargs = dict(kwargs)
        new_kwargs["return_when"] = asyncio.ALL_COMPLETED
        done_orig, pending_orig = await real_wait(fs, *args, **new_kwargs)
        ordered = _OrderedDone()
        rest: list[Any] = []
        for t in done_orig:
            if not t.cancelled() and t.exception() is not None:
                ordered.append(t)
            else:
                rest.append(t)
        ordered.extend(rest)
        return ordered, pending_orig

    async def _query_leader(address: str, **_kw: object) -> str | None:
        if address == "rejector:9001":
            return tainted
        return "winner:9001"

    async def _verify_redirect(address: str, **_kw: object) -> str | None:
        return address

    def _policy(address: str) -> bool:
        return "\n" not in address and "\r" not in address

    store = MemoryNodeStore(["rejector:9001", "winner:9001"])
    cluster = ClusterClient(store, timeout=10.0, redirect_policy=_policy)
    cluster._query_leader = AsyncMock(side_effect=_query_leader)
    cluster._verify_redirect = AsyncMock(side_effect=_verify_redirect)

    from unittest.mock import patch

    caplog.set_level(logging.WARNING, logger="dqliteclient.cluster")
    with patch("dqliteclient.cluster.asyncio.wait", _wait_for_all_then_order):
        leader = await asyncio.wait_for(cluster.find_leader(), timeout=2.0)
    assert leader == "winner:9001"

    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "dropped policy rejection during successful" in r.getMessage()
    ]
    assert matching, "WARN must fire on the both-states-set path"

    rendered = matching[0].getMessage()
    # CWE-117 pin: a regression dropping ``sanitize_for_log`` from
    # the WARN formatter would leave the embedded CRLF in the
    # rendered record — splitting it into two log lines at the
    # journald / syslog layer.
    assert "\n" not in rendered, (
        f"WARN message must NOT contain raw newline (CWE-117): {rendered!r}"
    )
    assert "\r" not in rendered, (
        f"WARN message must NOT contain raw carriage return (CWE-117): {rendered!r}"
    )
    # Positive evidence that sanitisation ran: the escaped form is
    # present (``sanitize_for_log`` replaces LF with ``\\n``).
    assert "\\n" in rendered or "INJECTED" not in rendered.split("\n")[0], (
        f"Expected sanitised escape of CRLF in WARN message: {rendered!r}"
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
