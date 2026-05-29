"""Pin: ``acquire()``'s exception arm re-checks ``_closed`` after the
``_reset_connection`` (ROLLBACK) yield.

close() does not take ``_lock``, so a concurrent close can flip
``_closed=True`` and drain the queue while the ROLLBACK is in flight;
without the re-check the conn would be queued into a drained pool and
orphaned. (The structurally-identical ``_release`` path is pinned
separately.)
"""

from __future__ import annotations

import asyncio
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
        self._protocol = MagicMock()
        self._protocol._writer = MagicMock()
        self._protocol._writer.transport = MagicMock()
        self._protocol._writer.transport.is_closing = lambda: False
        self._protocol._reader = MagicMock()
        self._protocol._reader.at_eof = lambda: False
        self._protocol.is_wire_coherent = True
        self.close_called = False

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    async def close(self) -> None:
        self.close_called = True
        self._protocol = None  # type: ignore[assignment]


def _make_pool() -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_acquire_exception_post_reset_closes_when_pool_closed_during_reset() -> None:
    """A concurrent close() during the ROLLBACK yield must route the conn
    through close + reservation release, not orphan it in a drained queue."""
    pool = _make_pool()
    pool._size = 1
    conn = _FakeConn()
    pool._pool.put_nowait(conn)  # type: ignore[arg-type]

    reset_can_finish = asyncio.Event()
    reset_entered = asyncio.Event()

    async def _slow_reset(c: Any) -> bool:
        reset_entered.set()
        await reset_can_finish.wait()
        return True

    pool._reset_connection = _slow_reset  # type: ignore[assignment]

    async def _user() -> None:
        async with pool.acquire():
            raise RuntimeError("user error")

    user_task = asyncio.create_task(_user())
    await reset_entered.wait()
    assert not user_task.done()

    close_task = asyncio.create_task(pool.close())
    for _ in range(10):
        await asyncio.sleep(0)
        if pool._closed:
            break
    assert pool._closed, "expected pool.close() to mark _closed=True"

    reset_can_finish.set()
    with pytest.raises(RuntimeError, match="user error"):
        await user_task
    await close_task

    # Closed via the post-reset re-check, not orphaned in a drained queue.
    assert conn.close_called is True
    assert conn._pool_released is True
