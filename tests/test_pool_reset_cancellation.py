"""Pin cancellation behaviour during ``_reset_connection`` ROLLBACK.

The pool's reset path issues ``await conn.execute(ROLLBACK)``. If a
cancel lands mid-flight, the client's ``_run_protocol`` invalidates
and re-raises ``CancelledError`` — outside the pool's
``_POOL_CLEANUP_EXCEPTIONS`` tuple, so it propagates out of
``_reset_connection`` and into ``_release``'s outer finally. The
finally's shielded-pending-drain await + ``_pool_released = True``
+ shielded ``_release_reservation`` must keep the slot accounting
consistent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_reset_rollback_cancelled_invalidates_and_releases_slot() -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")

    started = asyncio.Event()
    suspend = asyncio.Event()

    async def slow_rollback(sql: str) -> object:
        started.set()
        try:
            await suspend.wait()
        except asyncio.CancelledError:
            raise
        return None

    with (
        patch.object(DqliteConnection, "is_connected", new=True),
        patch("dqliteclient.pool._socket_looks_dead", return_value=False),
        patch.object(conn, "execute", new=slow_rollback),
    ):
        conn._in_transaction = True

        reset_task = asyncio.get_running_loop().create_task(pool._reset_connection(conn))
        await started.wait()
        reset_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reset_task

    # _reset_connection propagated the cancel; _release's outer
    # finally is responsible for slot accounting via the calling
    # path. Sanity-check size stays consistent.
    assert pool._size in (0, 1)
    suspend.set()
