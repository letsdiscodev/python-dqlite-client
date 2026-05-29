"""Pin: ``ConnectionPool.initialize()`` does not hold ``self._lock`` across the
parallel-create gather, so concurrent ``acquire()`` callers make progress."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self) -> None:
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol: MagicMock | None = MagicMock()
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self._protocol = None

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _pool_with_slow_factory(slow_seconds: float, *, min_size: int, max_size: int) -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)

    async def _factory(**kwargs: Any) -> _FakeConn:
        await asyncio.sleep(slow_seconds)
        return _FakeConn()

    cluster.connect = _factory
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=min_size,
        max_size=max_size,
        timeout=10.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_concurrent_acquire_does_not_park_on_initialize_lock() -> None:
    """A sibling acquire() scheduled while initialize() is mid-gather runs its
    own dial in parallel instead of parking on the init lock."""
    pool = _pool_with_slow_factory(0.5, min_size=3, max_size=10)

    async def _acquire_after_50ms() -> float:
        await asyncio.sleep(0.05)
        start = time.perf_counter()
        async with pool.acquire():
            pass
        return time.perf_counter() - start

    init_task = asyncio.create_task(pool.initialize())
    acquire_elapsed = await _acquire_after_50ms()
    await init_task

    # Post-fix ~0.5 s (own dial); pre-fix ~0.95 s (parked on init lock + dial).
    assert acquire_elapsed < 0.8, (
        f"sibling acquire took {acquire_elapsed:.3f} s while initialize "
        f"was running; expected under 0.8 s. The init lock must NOT be "
        f"held across the parallel-create gather."
    )

    await pool.close()
