"""Pin: ``_find_leader_impl``'s cached-leader fast path passes
``self._attempt_timeout`` to ``_verify_redirect``, not the default
``self._dial_timeout``.

The cached fast path is single-RTT followed by an optional single
verify-RTT — sequential, not nested. The ``_verify_redirect``
default of ``dial_timeout`` is sized for the parallel-sweep call
site (where the verify is nested inside an outer
``attempt_timeout``-bounded probe and using ``attempt_timeout``
again would let the worst case be 2 × ``attempt_timeout``). The
cached fast-path has no nesting to halve, so ``dial_timeout`` would
drop healthy-but-loaded redirect targets when operators size
``dial_timeout < attempt_timeout``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_cached_fast_path_verify_uses_attempt_timeout() -> None:
    """When the cached leader returns a redirect, the follow-up
    ``_verify_redirect`` must be called with ``timeout=self._attempt_timeout``
    so the cached path's single-RTT verify gets the same envelope as
    the original probe (rather than the smaller ``dial_timeout``)."""
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        dial_timeout=1.0,
        attempt_timeout=10.0,
    )
    # Plant a cached leader so the fast path runs.
    cluster._set_last_known_leader("cached:9001")
    # Cached node returns a redirect to a different address.
    cluster._query_leader = AsyncMock(return_value="redirect-target:9001")
    # Spy on _verify_redirect so we observe the timeout argument.
    spy = AsyncMock(return_value="redirect-target:9001")
    with patch.object(cluster, "_verify_redirect", spy):
        result = await cluster._find_leader_impl(trust_server_heartbeat=False)
    assert result == "redirect-target:9001"
    # The cached fast-path must pass attempt_timeout (10.0), NOT the
    # _verify_redirect default of dial_timeout (1.0).
    spy.assert_called_once()
    call_kwargs = spy.call_args.kwargs
    assert call_kwargs.get("timeout") == 10.0, (
        f"cached fast-path verify must use attempt_timeout (10.0); got "
        f"timeout={call_kwargs.get('timeout')!r}"
    )


@pytest.mark.asyncio
async def test_verify_redirect_default_is_dial_timeout() -> None:
    """Pin: ``_verify_redirect``'s default (``timeout=None``) falls
    back to ``self._dial_timeout`` so existing parallel-sweep call
    sites keep their nesting-cost halving.

    Observe the timeout via a monkeypatched ``asyncio.wait_for`` that
    captures the ``timeout`` kwarg. The earlier shape of this test
    asserted only that ``_query_leader`` was called once — a
    regression that swapped the default to ``attempt_timeout`` would
    keep the call-count unchanged and silently pass.
    """
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        dial_timeout=1.5,
        attempt_timeout=8.0,
    )
    cluster._query_leader = AsyncMock(return_value="leader:9001")  # self-confirm

    captured_timeouts: list[float | None] = []

    from dqliteclient import cluster as cluster_module

    real_wait_for = cluster_module.asyncio.wait_for  # type: ignore[attr-defined]

    async def _wait_for_capture(coro: Any, *, timeout: float | None) -> Any:
        captured_timeouts.append(timeout)
        return await real_wait_for(coro, timeout=timeout)

    with patch.object(cluster_module.asyncio, "wait_for", new=_wait_for_capture):  # type: ignore[attr-defined]
        result = await cluster._verify_redirect("leader:9001")
    assert result == "leader:9001"
    # Default timeout=None → effective_timeout = self._dial_timeout (1.5).
    # The capture pins the actual envelope passed to wait_for, not just
    # the call-count.
    assert captured_timeouts == [1.5], (
        f"_verify_redirect default must resolve to self._dial_timeout (1.5); "
        f"captured={captured_timeouts}"
    )
