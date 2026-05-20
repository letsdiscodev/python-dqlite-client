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
