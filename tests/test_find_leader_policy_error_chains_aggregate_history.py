"""When ``_find_leader_impl`` raises ``ClusterPolicyError`` on the
policy-rejection arm, the accumulated per-node transport history
(timeouts / refused / no-leader replies) must be chained on the
exception's ``__cause__`` — mirroring the no-policy aggregate arm
which chains via ``BaseExceptionGroup``.

Pre-fix the policy arm re-raised bare, dropping the forensic evidence
that distinguishes 'policy rejected the leader on an otherwise-healthy
cluster' from 'policy rejected the leader on a half-down cluster'.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import (
    ClusterPolicyError,
    DqliteConnectionError,
)
from dqliteclient.node_store import MemoryNodeStore


def test_policy_error_chains_aggregate_per_node_excs() -> None:
    """One probe raises ClusterPolicyError; two others fail transport-
    wise. The raised exception must be a ClusterPolicyError with the
    accumulated per-node failures on its __cause__ as a
    BaseExceptionGroup."""
    cc = ClusterClient(
        MemoryNodeStore(["a:9001", "b:9001", "c:9001"]),
        timeout=2.0,
        redirect_policy=lambda _addr: False,
    )

    async def _fake_query_leader(address: str, **_kw: object) -> str | None:
        if address == "a:9001":
            # Returns a redirect to a non-allowed target — triggers the
            # policy-error branch.
            return "leader:9001"
        # The other two nodes time out / refuse.
        raise DqliteConnectionError(f"{address}: down")

    with (
        patch.object(cc, "_query_leader", side_effect=_fake_query_leader),
        pytest.raises(ClusterPolicyError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    cause = exc_info.value.__cause__
    assert cause is not None, (
        "policy-error arm must chain the accumulated per-node failures "
        "via __cause__; bare re-raise drops forensic context"
    )
    assert isinstance(cause, BaseExceptionGroup), (
        f"expected a BaseExceptionGroup chain (mirrors the aggregate arm "
        f"at cluster.py:1107-1110); got {type(cause).__name__}"
    )
    chained = list(cause.exceptions)
    assert len(chained) >= 2, (
        f"expected at least the two transport failures from b:9001 and c:9001; got {chained!r}"
    )


def test_policy_error_with_single_per_node_exc_chains_directly() -> None:
    """Single non-policy per-node exception: chained on ``__cause__``
    directly (no BaseExceptionGroup wrap) — mirrors the aggregate arm's
    single-exception branch at cluster.py:1111-1112."""
    cc = ClusterClient(
        MemoryNodeStore(["a:9001", "b:9001"]),
        timeout=2.0,
        redirect_policy=lambda _addr: False,
    )

    async def _fake_query_leader(address: str, **_kw: object) -> str | None:
        if address == "a:9001":
            return "leader:9001"
        raise DqliteConnectionError(f"{address}: down")

    with (
        patch.object(cc, "_query_leader", side_effect=_fake_query_leader),
        pytest.raises(ClusterPolicyError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    cause = exc_info.value.__cause__
    assert cause is not None
    # Single transport failure: chained narrow rather than wrapped.
    assert isinstance(cause, DqliteConnectionError)


def test_policy_error_with_no_other_failures_keeps_bare_raise() -> None:
    """When the policy rejection is the ONLY surface — no transport
    failures, no no-leader-known accumulation — the bare re-raise is
    the right shape (no spurious __cause__)."""
    cc = ClusterClient(
        MemoryNodeStore(["a:9001"]),
        timeout=2.0,
        redirect_policy=lambda _addr: False,
    )

    async def _fake_query_leader(_address: str, **_kw: object) -> str | None:
        return "leader:9001"

    with (
        patch.object(cc, "_query_leader", side_effect=_fake_query_leader),
        pytest.raises(ClusterPolicyError) as exc_info,
    ):
        asyncio.run(cc.find_leader())

    assert exc_info.value.__cause__ is None
