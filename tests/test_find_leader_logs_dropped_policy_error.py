"""Parallel-sweep policy-rejection observability: dropped-rejection WARN,
all-rejected cache invalidation, and ValueError fast-path translation."""

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
    """Inspection pin: when both ``winning_address`` and ``policy_error``
    are set, ``_find_leader_impl`` emits the dropped-rejection WARN naming
    both, sanitised via ``sanitize_for_log`` (CWE-117)."""
    import inspect

    src = inspect.getsource(ClusterClient._find_leader_impl)
    assert "dropped policy rejection during successful" in src
    assert "sanitize_for_log" in src, (
        "WARN log must sanitise the policy_error and winning_address "
        "(CWE-117); the substring asserts the wrapping is present"
    )
    assert "if policy_error is not None:" in src
    assert "winning_address is not None:" in src


@pytest.mark.asyncio
async def test_parallel_sweep_all_policy_rejected_invalidates_cache() -> None:
    """When every probe redirects to a policy-rejected target, the sweep
    clears the leader cache before raising so the next call's fast path
    does not waste an RTT on the stale entry."""
    addresses = ["a:9001", "b:9001"]
    store = MemoryNodeStore(addresses)

    def policy(address: str) -> bool:
        return False  # reject everything

    cluster = ClusterClient(store, timeout=10.0, redirect_policy=policy)
    cluster._set_last_known_leader("stale.example.com:9001")
    assert cluster._get_last_known_leader() == "stale.example.com:9001"

    async def _query_leader(address: str, **_kw: object) -> str:
        return "redirect-target:9001"

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
    target and another self-confirms as leader, the WARN fires naming both,
    routed through ``sanitize_for_log`` (CWE-117).

    Deterministic via monkey-patching ``asyncio.wait`` to return ``done``
    rejector-first, avoiding the hash-order flakiness of the real race."""
    real_wait = asyncio.wait

    class _OrderedDone(list[asyncio.Task[object]]):
        pass

    async def _wait_for_all_then_order(
        fs: Any, *args: Any, **kwargs: Any
    ) -> tuple[_OrderedDone, set[Any]]:
        # ALL_COMPLETED so both probes resolve in one iteration, rejector
        # first via insertion order, reproducing the race deterministically.
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
    rendered = matching[0].getMessage()
    assert "rejected-target:9001" in rendered, (
        f"WARN must name the rejected redirect target: got {rendered!r}"
    )
    assert "winner:9001" in rendered, f"WARN must name the winning leader address: got {rendered!r}"


@pytest.mark.asyncio
async def test_dropped_policy_warn_sanitises_crlf_tainted_address(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CWE-117 pin: the WARN routes the policy-error text through
    ``sanitize_for_log`` so a CRLF-tainted redirect target cannot split
    the log record."""
    # Tainted address a hostile peer could smuggle via a redirect target.
    tainted = "evil.example.com:9001\r\nINJECTED log line"

    real_wait = asyncio.wait

    class _OrderedDone(list[asyncio.Task[object]]):
        pass

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
    assert "\n" not in rendered, (
        f"WARN message must NOT contain raw newline (CWE-117): {rendered!r}"
    )
    assert "\r" not in rendered, (
        f"WARN message must NOT contain raw carriage return (CWE-117): {rendered!r}"
    )
    assert "\\n" in rendered or "INJECTED" not in rendered.split("\n")[0], (
        f"Expected sanitised escape of CRLF in WARN message: {rendered!r}"
    )


def test_fast_path_catches_value_error_from_malformed_address() -> None:
    """Inspection pin: the fast-path catch tuple includes ``ValueError`` so
    a malformed address from a third-party NodeStore is translated."""
    import inspect

    src = inspect.getsource(ClusterClient._find_leader_impl)
    assert "ValueError" in src, (
        "Fast-path catch must include ``ValueError`` to translate "
        "malformed addresses from third-party NodeStores"
    )
