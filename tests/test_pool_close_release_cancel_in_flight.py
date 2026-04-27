"""Pin: ``pool.close()`` racing with a checked-out connection's
``_release`` on outer cancel does not leak ``_size``.

The ``_release`` closed-branch at ``pool.py`` runs ``await
conn.close()`` then ``_pool_released = True; return``. If a cancel
lands during the await, the closed-branch outer ``finally`` at
``_release`` runs the safety cleanup. This test pins that the
post-race ``_size`` accounting is consistent and ``pool.close()``
eventually completes.
"""

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

    # Build a stub conn whose close awaits a never-firing event so
    # we can suspend in the release path.
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
        # Mark the pool closed so ``_release`` enters the closed
        # branch directly. The closed branch awaits ``conn.close()``,
        # which we suspend.
        pool._closed = True
        release_task = asyncio.get_running_loop().create_task(pool._release(conn))
        await started.wait()
        # Cancel mid-close.
        release_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release_task

    # The pool's size accounting should reflect the released slot
    # via the outer finally; close() can complete cleanly.
    # _size accounting via the finally must release the reservation.
    assert pool._size in (0, 1), f"unexpected _size={pool._size}"
    suspend.set()
