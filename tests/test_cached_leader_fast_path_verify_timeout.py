"""Cached-leader fast path passes ``attempt_timeout`` to ``_verify_redirect``.

The fast path is sequential (single probe + single verify), not nested, so the
``_verify_redirect`` default of ``dial_timeout`` (sized to halve the parallel-sweep
nesting cost) would drop healthy targets when ``dial_timeout < attempt_timeout``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_cached_fast_path_verify_uses_attempt_timeout() -> None:
    """A cached-leader redirect verifies with ``timeout=attempt_timeout``."""
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        dial_timeout=1.0,
        attempt_timeout=10.0,
    )
    cluster._set_last_known_leader("cached:9001")
    cluster._query_leader = AsyncMock(return_value="redirect-target:9001")
    spy = AsyncMock(return_value="redirect-target:9001")
    with patch.object(cluster, "_verify_redirect", spy):
        result = await cluster._find_leader_impl(trust_server_heartbeat=False)
    assert result == "redirect-target:9001"
    spy.assert_called_once()
    call_kwargs = spy.call_args.kwargs
    assert call_kwargs.get("timeout") == 10.0, (
        f"cached fast-path verify must use attempt_timeout (10.0); got "
        f"timeout={call_kwargs.get('timeout')!r}"
    )


@pytest.mark.asyncio
async def test_verify_redirect_default_is_dial_timeout() -> None:
    """``_verify_redirect``'s default (``timeout=None``) resolves to
    ``self._dial_timeout``, preserving the parallel-sweep nesting-cost halving."""
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
        dial_timeout=1.5,
        attempt_timeout=8.0,
    )
    cluster._query_leader = AsyncMock(return_value="leader:9001")

    captured_timeouts: list[float | None] = []

    from dqliteclient import cluster as cluster_module

    real_timeout = cluster_module.asyncio.timeout  # type: ignore[attr-defined]

    def _timeout_capture(delay: float | None) -> Any:
        captured_timeouts.append(delay)
        return real_timeout(delay)

    with patch.object(cluster_module.asyncio, "timeout", new=_timeout_capture):  # type: ignore[attr-defined]
        result = await cluster._verify_redirect("leader:9001")
    assert result == "leader:9001"
    assert captured_timeouts == [1.5], (
        f"_verify_redirect default must resolve to self._dial_timeout (1.5); "
        f"captured={captured_timeouts}"
    )
