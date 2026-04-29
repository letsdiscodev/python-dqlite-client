"""Pin: pool ``_release`` re-checks ``_closed`` after the
``_reset_connection`` yield.

``pool.close()`` does not take ``_lock``, so it may flip
``_closed=True`` and drain the queue while ``_release`` is
suspended awaiting the ROLLBACK. Without the post-reset re-check,
the conn would be put_nowait'd into a drained queue and orphaned
(acquire() raises "Pool is closed", the conn is unreachable).

Existing tests cover the connect-site analogous race; the
``_release`` site post-reset re-check was untested. A future
refactor that drops the re-check would silently re-introduce the
orphan-into-drained-queue bug.
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
    """``_release`` must observe a concurrent ``pool.close()`` that
    landed during the ``_reset_connection`` yield, route the conn
    through close + reservation release, and not orphan it in a
    drained queue."""
    pool = _make_pool()
    pool._size = 1
    conn = _FakeConn()

    # Block _reset_connection on a controllable event so we can
    # interleave a pool.close() before the ROLLBACK returns.
    reset_can_finish = asyncio.Event()
    reset_entered = asyncio.Event()

    async def _slow_reset(c: Any) -> bool:
        reset_entered.set()
        await reset_can_finish.wait()
        return True  # ROLLBACK "succeeded"

    pool._reset_connection = _slow_reset  # type: ignore[assignment]

    # Start _release on a background task; it parks inside _slow_reset.
    release_task = asyncio.create_task(pool._release(conn))  # type: ignore[arg-type]
    await reset_entered.wait()
    assert not release_task.done()

    # Concurrently close the pool. The post-reset re-check must
    # observe _closed=True after we let _slow_reset return.
    close_task = asyncio.create_task(pool.close())
    # Give close() a tick to mark _closed=True before we let _slow_reset finish.
    for _ in range(5):
        await asyncio.sleep(0)
        if pool._closed:
            break
    assert pool._closed, "expected pool.close() to mark _closed=True"

    reset_can_finish.set()
    await release_task
    await close_task

    # The conn must have been close()'d (post-reset re-check
    # branched into the close+release path), not orphaned in the
    # drained queue.
    assert conn.close_called is True
    assert conn._pool_released is True
