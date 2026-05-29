"""Pin: ``_release`` re-checks ``_closed`` after the ``_reset_connection`` yield.

pool.close() does not take _lock, so it may flip _closed=True and drain the queue
while _release awaits the ROLLBACK. Without the re-check the conn is put_nowait'd
into a drained queue and orphaned.
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
async def test_release_post_reset_closes_conn_when_pool_closed_during_reset() -> None:
    """A concurrent pool.close() during the reset yield routes the conn through
    close + reservation release rather than orphaning it in a drained queue."""
    pool = _make_pool()
    pool._size = 1
    conn = _FakeConn()

    # Block _reset_connection so we can interleave pool.close() before ROLLBACK returns.
    reset_can_finish = asyncio.Event()
    reset_entered = asyncio.Event()

    async def _slow_reset(c: Any) -> bool:
        reset_entered.set()
        await reset_can_finish.wait()
        return True  # ROLLBACK "succeeded"

    pool._reset_connection = _slow_reset  # type: ignore[assignment]

    release_task = asyncio.create_task(pool._release(conn))  # type: ignore[arg-type]
    await reset_entered.wait()
    assert not release_task.done()

    close_task = asyncio.create_task(pool.close())
    # Let close() mark _closed=True before _slow_reset finishes.
    for _ in range(5):
        await asyncio.sleep(0)
        if pool._closed:
            break
    assert pool._closed, "expected pool.close() to mark _closed=True"

    reset_can_finish.set()
    await release_task
    await close_task

    # close()'d via the re-check's close+release path, not orphaned in the queue.
    assert conn.close_called is True
    assert conn._pool_released is True
