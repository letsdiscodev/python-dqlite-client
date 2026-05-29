"""Pin: a cancel landing during ``_release``'s closed-branch ``conn.close()``
must not leak ``_size`` — the outer finally releases the reservation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_pool_close_release_cancel_inflight_keeps_size_consistent() -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=2)
    pool._size = 1

    # Stub conn whose close awaits a never-firing event to suspend in _release.
    conn = DqliteConnection("localhost:9001")
    suspend = asyncio.Event()
    started = asyncio.Event()

    async def fake_close() -> None:
        started.set()
        try:
            await suspend.wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            raise

    with (
        patch.object(DqliteConnection, "is_connected", new=True),
        patch.object(conn, "close", new=fake_close),
    ):
        # Closed pool: _release enters the closed branch and awaits conn.close().
        pool._closed = True
        release_task = asyncio.get_running_loop().create_task(pool._release(conn))
        await started.wait()
        release_task.cancel()  # cancel mid-close
        with pytest.raises(asyncio.CancelledError):
            await release_task

    # The outer finally must release the reservation slot.
    assert pool._size in (0, 1), f"unexpected _size={pool._size}"
    suspend.set()
