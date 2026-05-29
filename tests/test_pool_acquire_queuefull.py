"""QueueFull during ``acquire()``'s BaseException cleanup arm must close the
won connection and decrement ``_size``, rather than silently dropping it
(which would leak a live reader task + socket and shrink pool capacity).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._address = "localhost:9001"
        self._in_transaction = False
        self._tx_owner = None
        self._pool_released = False
        self._protocol = MagicMock()
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False
        self.close_called = False
        self.close_effective = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        if self._pool_released or self._protocol is None:
            return
        self.close_effective = True
        self._protocol = None  # type: ignore[assignment]

    async def execute(self, sql: str, params: Any = None) -> tuple[int, int]:
        return (0, 0)


def _make_pool_with_fake_cluster(
    *,
    min_size: int = 0,
    max_size: int = 2,
) -> tuple[ConnectionPool, list[_FakeConn]]:
    created: list[_FakeConn] = []

    async def _default_connect(**kwargs: Any) -> _FakeConn:
        c = _FakeConn(name=f"c{len(created)}")
        created.append(c)
        return c

    cluster = MagicMock(spec=ClusterClient)
    cluster.connect = _default_connect
    pool = ConnectionPool(
        addresses=["localhost:9001"],
        min_size=min_size,
        max_size=max_size,
        timeout=1.0,
        cluster=cluster,
    )
    return pool, created


@pytest.mark.asyncio
async def test_queuefull_on_reinsert_closes_won_conn_and_shrinks_size() -> None:
    """Force the cleanup arm to see a won get_task plus a QueueFull on
    put_nowait, then assert the won conn is closed and ``_size`` decremented.
    The race is too narrow to hit naturally, so ``asyncio.wait`` is patched
    to drive it deterministically."""
    pool, created = _make_pool_with_fake_cluster(max_size=1)

    # Occupy the one slot so the next acquire reaches the wait arm.
    blocking = await pool.acquire().__aenter__()
    assert blocking is created[0]
    size_before = pool._size
    assert size_before == 1

    # Patch asyncio.wait so the waiter sees a done get_task then is
    # cancelled, and make put_nowait raise QueueFull in the same arm.
    import dqliteclient.pool as pool_mod

    real_wait = pool_mod.asyncio.wait  # type: ignore[attr-defined]
    original_put_nowait = pool._pool.put_nowait

    wait_called = []

    async def fake_wait(tasks, *, timeout=None, return_when):
        wait_called.append(1)
        # Unblock and resolve get_task by queueing the held conn.
        original_put_nowait(blocking)
        await real_wait(tasks, timeout=0.5, return_when=return_when)
        # Make the cleanup arm's re-insertion fail, then cancel.
        pool._pool.put_nowait = _raise_queue_full  # type: ignore[assignment]
        raise asyncio.CancelledError

    def _raise_queue_full(_conn: object) -> None:
        raise asyncio.QueueFull

    pool_mod.asyncio.wait = fake_wait  # type: ignore[attr-defined]
    cm = pool.acquire()
    try:
        await cm.__aenter__()
    except asyncio.CancelledError:
        pass
    except BaseException as e:  # pragma: no cover - defensive
        pool_mod.asyncio.wait = real_wait  # type: ignore[attr-defined]
        pool._pool.put_nowait = original_put_nowait
        raise AssertionError(f"expected CancelledError, got {type(e).__name__}: {e}") from e
    else:  # pragma: no cover - defensive
        pool_mod.asyncio.wait = real_wait  # type: ignore[attr-defined]
        pool._pool.put_nowait = original_put_nowait
        raise AssertionError("expected acquire to be cancelled, but it returned")
    pool_mod.asyncio.wait = real_wait  # type: ignore[attr-defined]
    pool._pool.put_nowait = original_put_nowait
    assert wait_called, "fake_wait monkeypatch did not apply"

    assert created[0].close_called, (
        "QueueFull cleanup path must call close() on the won conn, not silently drop the reference"
    )
    assert pool._size == size_before - 1, (
        f"QueueFull cleanup must decrement _size (was {size_before}, now {pool._size})"
    )

    await pool.close()
