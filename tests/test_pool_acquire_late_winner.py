"""``ConnectionPool._put_back_or_release_late_winner`` is the recovery
hook for two race paths in ``acquire``: an outer cancel that arrives
just after a sibling ``_release`` puts a conn on the queue, and the
post-wait demux's else-arm where a timeout snapshot raced a winning
``get_task``. Both paths route through this helper so the conn is
not silently dropped.

The helper has two arms:
- queue not full → ``put_nowait`` and the conn is reusable.
- queue full → close + release reservation (invariant-violation
  recovery).

Pin both arms.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_late_winner_put_back_when_queue_has_room() -> None:
    """The happy path: queue has room, conn is put back and the
    pool's free-conn counter reflects it."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=4)
    fake_conn = MagicMock()

    await pool._put_back_or_release_late_winner(fake_conn)

    assert pool._pool.qsize() == 1
    queued = pool._pool.get_nowait()
    assert queued is fake_conn


@pytest.mark.asyncio
async def test_late_winner_close_and_release_on_queue_full() -> None:
    """The recovery path: queue is full, conn is closed and the
    reservation released so the pool shrinks rather than leaks."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=1)
    # Saturate the queue so put_nowait raises QueueFull.
    filler = MagicMock()
    pool._pool.put_nowait(filler)

    fake_conn = MagicMock()
    fake_conn.close = AsyncMock()

    with patch.object(pool, "_release_reservation", new=AsyncMock()) as rel:
        await pool._put_back_or_release_late_winner(fake_conn)

    fake_conn.close.assert_awaited_once()
    rel.assert_awaited_once()
