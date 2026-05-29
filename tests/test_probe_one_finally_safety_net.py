"""``_probe_one``'s outer ``finally`` releases the semaphore slot when an
unexpected exception (outside the catch tuple) skips the explicit release;
without it the sweep leaks slots and deadlocks under cancellation. The
``sem_acquired`` flag prevents double-release after the explicit release."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_probe_one_outer_finally_safety_net_present_in_source() -> None:
    """Source pin: the ``sem_acquired`` flag + outer-finally release pair
    survive refactors (runtime check is unreliable — the slot state is
    internal to ``_find_leader_impl`` and not observable publicly)."""
    import inspect

    impl = inspect.getsource(ClusterClient._find_leader_impl)

    assert "sem_acquired = False" in impl
    assert "sem_acquired = True" in impl
    # Explicit release must clear the flag so the safety-net can't double-release.
    assert "sem_acquired = False" in impl.split("sem_acquired = True", 1)[1]
    assert "if sem_acquired:" in impl, "Outer finally safety-net must guard release on sem_acquired"


@pytest.mark.asyncio
async def test_probe_one_unexpected_exception_propagates() -> None:
    """An unexpected exception class from a probe propagates past find_leader."""
    cluster = ClusterClient(
        MemoryNodeStore(["a:9001"]),
        timeout=5.0,
        concurrent_leader_conns=1,
    )

    async def _explode(_addr: str, **_kw: Any) -> str | None:
        raise MemoryError("synthetic unexpected exception")

    with (
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_explode)),
        pytest.raises(MemoryError, match="synthetic"),
    ):
        await asyncio.wait_for(cluster.find_leader(), timeout=2.0)


@pytest.mark.asyncio
async def test_probe_one_safety_net_releases_slot_after_unexpected_exception() -> None:
    """With a single slot, a leaked slot after an unexpected exception would
    deadlock the next find_leader; wait_for bounds the regression."""
    cluster = ClusterClient(
        MemoryNodeStore(["only:9001"]),
        timeout=5.0,
        concurrent_leader_conns=1,
    )

    call_count = 0

    async def _bomb_first_then_succeed(_addr: str, **_kw: Any) -> str | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Trip the safety net via an exception outside the catch tuple.
            raise MemoryError("synthetic unexpected exception")
        return "only:9001"

    with patch.object(cluster, "_query_leader", AsyncMock(side_effect=_bomb_first_then_succeed)):
        with pytest.raises(MemoryError, match="synthetic"):
            await asyncio.wait_for(cluster.find_leader(), timeout=2.0)

        # A leaked slot would hang this acquire forever; wait_for surfaces it.
        leader = await asyncio.wait_for(cluster.find_leader(), timeout=2.0)
        assert leader == "only:9001"


@pytest.mark.asyncio
async def test_probe_one_safety_net_does_not_double_release_after_phase1() -> None:
    """After the explicit phase-1 release, ``sem_acquired = False`` must keep the
    outer finally from double-releasing on a phase-2 unexpected exception. Pin is
    "release called exactly once" since Semaphore.release() never raises on over-release."""
    cluster = ClusterClient(
        MemoryNodeStore(["primary:9001"]),
        timeout=5.0,
        concurrent_leader_conns=1,
    )

    release_calls: list[int] = []
    real_semaphore_cls = asyncio.Semaphore

    class _CountingSemaphore(real_semaphore_cls):  # type: ignore[misc, valid-type]
        def release(self, *args: Any, **kwargs: Any) -> None:
            release_calls.append(1)
            super().release(*args, **kwargs)

    async def _redirect_response(_addr: str, **_kw: Any) -> str | None:
        # Different address so redirect-verify fires after phase-1 release.
        return "redirect-target:9001"

    async def _verify_explodes(_addr: str, **_kw: Any) -> str | None:
        raise MemoryError("synthetic phase-2 unexpected exception")

    with (
        patch.object(asyncio, "Semaphore", _CountingSemaphore),
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_redirect_response)),
        patch.object(cluster, "_verify_redirect", AsyncMock(side_effect=_verify_explodes)),
        pytest.raises(MemoryError, match="synthetic phase-2"),
    ):
        await asyncio.wait_for(cluster.find_leader(), timeout=2.0)

    # Dropping ``sem_acquired = False`` after the explicit release surfaces as 2.
    assert release_calls == [1], (
        f"Outer finally must skip release after explicit phase-1 "
        f"release flipped sem_acquired to False; got {len(release_calls)} "
        f"release() calls (expected exactly 1)"
    )
