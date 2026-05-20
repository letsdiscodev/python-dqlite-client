"""Pin: ``_probe_one``'s outer ``finally`` safety-net releases the
semaphore slot when an unexpected exception (one outside the
narrow catch tuple) skips the explicit release between phase 1
and phase 2.

The manual-acquire / explicit-release / outer-finally idiom relies
on a flag (``sem_acquired``) to avoid double-release. The
load-bearing arm is the safety-net firing when the explicit release
was skipped — e.g. a ``MemoryError`` / ``KeyError`` from a
programmer-bug path, or any class not in
``(DqliteConnectionError, ProtocolError, OperationalError,
OSError, ValueError)``.

Without the safety-net, an unexpected exception in any probe
leaks the slot. Under cancellation pressure, the sweep deadlocks
when the slot count drops to zero. The existing
``test_parallel_sweep_releases_semaphore_before_verify`` covers
only the explicit-release before phase 2.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def test_probe_one_outer_finally_safety_net_present_in_source() -> None:
    """Inspection pin: ``_probe_one``'s outer ``finally`` carries the
    ``if sem_acquired: semaphore.release()`` safety-net.

    Verifying the safety-net via runtime test is unreliable here
    because the sweep cancels sibling probes the moment any
    unexpected exception escapes ``_probe_one`` (gather's
    return_exceptions=False default). The slot's release-vs-leak
    state is internal to ``_find_leader_impl`` and not observable
    via the public surface. The inspection pin asserts the load-
    bearing source-level invariant directly: the ``sem_acquired``
    flag + outer-finally release pair survive future refactors.

    A regression that drops the outer ``finally`` would leak a
    semaphore slot per unexpected-exception probe failure; under
    cancellation pressure, the sweep would deadlock when the slot
    count drops to zero.
    """
    import inspect

    impl = inspect.getsource(ClusterClient._find_leader_impl)

    # The flag must be initialised to False before acquire().
    assert "sem_acquired = False" in impl
    # The flag must flip to True after a successful acquire.
    assert "sem_acquired = True" in impl
    # An explicit release must clear the flag so the safety-net
    # cannot double-release after the legitimate post-phase-1 release.
    assert "sem_acquired = False" in impl.split("sem_acquired = True", 1)[1]
    # The outer finally must guard release on the flag.
    assert "if sem_acquired:" in impl, "Outer finally safety-net must guard release on sem_acquired"


@pytest.mark.asyncio
async def test_probe_one_unexpected_exception_propagates() -> None:
    """Documentation-style pin: an unexpected exception class from
    a probe propagates past ``find_leader``. This documents WHY the
    outer-finally safety-net exists — without it, the slot would
    leak on this path."""
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
    """Behavioural pin for the outer-finally safety-net.

    ``concurrent_leader_conns=1`` means a single slot governs every
    probe in the sweep. If the safety-net failed to release after an
    unexpected-class exception (``MemoryError`` here), the next
    ``find_leader`` would deadlock on ``await semaphore.acquire()``
    because the slot would never come back to the pool.

    ``asyncio.wait_for(..., timeout=2.0)`` bounds the regression mode
    so a failing test fails fast rather than hanging the suite.
    """
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
            # First probe: trip the safety net via an unexpected
            # exception class (outside the narrow catch tuple).
            raise MemoryError("synthetic unexpected exception")
        # Subsequent probes: return the address as its own leader so
        # the sweep completes without a redirect-verify round trip.
        return "only:9001"

    with patch.object(cluster, "_query_leader", AsyncMock(side_effect=_bomb_first_then_succeed)):
        with pytest.raises(MemoryError, match="synthetic"):
            await asyncio.wait_for(cluster.find_leader(), timeout=2.0)

        # Critical pin: if the safety-net leaked the only slot, the
        # next call's ``await semaphore.acquire()`` would hang
        # forever. ``wait_for`` surfaces a TimeoutError on regression.
        leader = await asyncio.wait_for(cluster.find_leader(), timeout=2.0)
        assert leader == "only:9001"


@pytest.mark.asyncio
async def test_probe_one_safety_net_does_not_double_release_after_phase1() -> None:
    """Twin behavioural pin: after the explicit phase-1 release, the
    ``sem_acquired = False`` reset must keep the outer finally from
    firing a second release.

    Drive a phase-2 unexpected exception by patching
    ``_verify_redirect`` to raise ``MemoryError`` after
    ``_query_leader`` returned a redirect target. The explicit
    release ran (flag flipped to False); the outer finally must skip
    its release branch. We instrument the semaphore via a wrapper
    that counts ``release()`` calls — exactly one is allowed for the
    explicit phase-1 release; a double-release would be two.

    ``asyncio.Semaphore.release()`` does not raise on over-release —
    it silently bumps the counter past the initial value (per
    reviewer note). The pin is therefore "release was called exactly
    once" rather than "ValueError raised".
    """
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
        # Hand back a DIFFERENT address so the redirect-verify path
        # fires after the explicit phase-1 release.
        return "redirect-target:9001"

    async def _verify_explodes(_addr: str, **_kw: Any) -> str | None:
        # ``MemoryError`` is OUTSIDE the narrow catch tuple inside
        # ``_probe_one`` so it propagates through the outer finally.
        raise MemoryError("synthetic phase-2 unexpected exception")

    with (
        patch.object(asyncio, "Semaphore", _CountingSemaphore),
        patch.object(cluster, "_query_leader", AsyncMock(side_effect=_redirect_response)),
        patch.object(cluster, "_verify_redirect", AsyncMock(side_effect=_verify_explodes)),
        pytest.raises(MemoryError, match="synthetic phase-2"),
    ):
        await asyncio.wait_for(cluster.find_leader(), timeout=2.0)

    # Phase 1 fired exactly one release; the outer finally must NOT
    # have double-released. Regression dropping ``sem_acquired = False``
    # after the explicit release would surface as 2.
    assert release_calls == [1], (
        f"Outer finally must skip release after explicit phase-1 "
        f"release flipped sem_acquired to False; got {len(release_calls)} "
        f"release() calls (expected exactly 1)"
    )
