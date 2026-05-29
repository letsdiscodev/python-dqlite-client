"""``_put_back_or_release_late_winner`` re-checks ``self._closed`` before
``put_nowait``: a cancel-during-acquire arriving after the pool's drain completed
would otherwise park the conn in a closed queue no one re-drains, leaking the socket."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_put_back_late_winner_into_closed_pool_closes_conn_and_releases_reservation() -> None:
    """With ``self._closed = True``, the helper must close the conn AND release the
    reservation; the queue must stay empty."""
    pool = ConnectionPool.__new__(ConnectionPool)
    import asyncio
    import threading

    pool._closed = True
    pool._pool = asyncio.Queue()
    pool._size = 1
    pool._lock = asyncio.Lock()
    pool._closed_event = asyncio.Event()
    pool._closed_event.set()
    pool._drain_complete = True
    pool._max_size = 2
    pool._signal_state_change = MagicMock()

    conn = MagicMock()
    conn.close = AsyncMock()
    conn._lock_owner = threading.get_ident()

    await pool._put_back_or_release_late_winner(conn)

    conn.close.assert_awaited_once()
    assert pool._pool.empty()
    assert pool._size == 0


@pytest.mark.asyncio
async def test_put_back_late_winner_into_open_pool_returns_conn_to_queue() -> None:
    """On an OPEN pool, the helper puts the conn back into the queue."""
    pool = ConnectionPool.__new__(ConnectionPool)
    import asyncio
    import threading

    pool._closed = False
    pool._pool = asyncio.Queue(maxsize=4)
    pool._size = 1
    pool._lock = asyncio.Lock()
    pool._closed_event = asyncio.Event()
    pool._drain_complete = False
    pool._max_size = 2
    pool._signal_state_change = MagicMock()

    conn = MagicMock()
    conn.close = AsyncMock()
    conn._lock_owner = threading.get_ident()

    await pool._put_back_or_release_late_winner(conn)

    conn.close.assert_not_awaited()
    assert pool._pool.qsize() == 1
    assert pool._size == 1
