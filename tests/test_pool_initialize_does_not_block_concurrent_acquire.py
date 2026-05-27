"""Pin: ``ConnectionPool.initialize()`` does NOT hold ``self._lock``
across the parallel-create gather, so concurrent ``pool.acquire()``
callers can make progress during the warm-up dial.

The prior shape wrapped the entire warm-up — including
``await asyncio.gather(*create_tasks)`` whose tasks each ran a
multi-second find-leader sweep + TCP dial + handshake — inside a
single ``async with self._lock:`` block. Concurrent ``acquire()``
callers parked on the same lock just to reserve a ``_size`` slot;
they could not run their own dial in parallel with the warm-up.

The post-fix 3-phase shape (bookkeeping under lock / gather without
lock / publish under lock) lets sibling ``acquire()`` callers
proceed at Phase B. Mirrors the ``psycopg_pool.AsyncConnectionPool``
``open()`` / ``wait()`` + ``_pool_full_event`` discipline (the 1:1
prior-art precedent identified in the issue file's prior-art
section).
"""

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
    """A ``pool.acquire()`` scheduled while ``pool.initialize()`` is
    mid-gather must not park on the init lock for the full warm-up
    duration. Under the post-fix shape, the acquire runs its own dial
    in parallel with the warm-up gather.

    Concrete test: warm-up dials take 0.5 s each (sleep). Schedule
    ``initialize()`` then 50 ms later schedule a sibling
    ``acquire()``. The sibling must complete inside ~0.6 s
    (= its own 0.5 s dial + scheduling slack), NOT inside ~1.0 s
    (= 0.5 s waiting for init lock release + 0.5 s of its own dial).
    """
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

    # Post-fix bound: ~0.5 s (just the sibling's own dial). Pre-fix
    # bound: ~0.95 s (~0.45 s parked on init lock + ~0.5 s own dial).
    # Pin at 0.8 s to leave generous slack for CI variance while
    # still catching the pre-fix regression.
    assert acquire_elapsed < 0.8, (
        f"sibling acquire took {acquire_elapsed:.3f} s while initialize "
        f"was running; expected under 0.8 s. The init lock must NOT be "
        f"held across the parallel-create gather."
    )

    await pool.close()
