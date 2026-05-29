"""Pin both arms of ``ConnectionPool._put_back_or_release_late_winner``:
queue-not-full puts the conn back; queue-full closes it and releases the
reservation. Either way the late-winner conn is never silently dropped.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_late_winner_put_back_when_queue_has_room() -> None:
    """Queue has room: conn is put back."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=4)
    fake_conn = MagicMock()

    await pool._put_back_or_release_late_winner(fake_conn)

    assert pool._pool.qsize() == 1
    queued = pool._pool.get_nowait()
    assert queued is fake_conn


@pytest.mark.asyncio
async def test_late_winner_close_and_release_on_queue_full() -> None:
    """Queue full: conn is closed and the reservation released."""
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
