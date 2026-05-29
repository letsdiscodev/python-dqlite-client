"""The policy-rejection ``ClusterPolicyError`` must chain accumulated
per-node transport failures on ``__cause__`` (like the no-policy aggregate
arm) so a healthy vs half-down cluster stays distinguishable."""

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
    """One probe triggers policy rejection, two fail transport-wise: the
    ClusterPolicyError must chain both failures via BaseExceptionGroup."""
    cc = ClusterClient(
        MemoryNodeStore(["a:9001", "b:9001", "c:9001"]),
        timeout=2.0,
        redirect_policy=lambda _addr: False,
    )

    async def _fake_query_leader(address: str, **_kw: object) -> str | None:
        if address == "a:9001":
            return "leader:9001"  # redirect to a non-allowed target
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
    """A single non-policy per-node exception is chained on ``__cause__``
    directly, with no BaseExceptionGroup wrap."""
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
    assert isinstance(cause, DqliteConnectionError)


def test_policy_error_with_no_other_failures_keeps_bare_raise() -> None:
    """A policy rejection with no other failures re-raises bare, with no
    spurious __cause__."""
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
