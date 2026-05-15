"""Pin: ``_put_back_or_release_late_winner`` re-checks ``self._closed``
before ``put_nowait`` so a cancel-during-acquire after the pool's
drain has completed does not leak the conn into a queue no one will
re-drain.

Race scenario the fix closes:

1. Task A is in ``acquire()``, parked in ``asyncio.wait`` inside
   ``_pool.get()``.
2. Task B (sibling) holds a conn, calls ``_release()``. ``_release``
   re-checks ``_closed`` (False), runs ``put_nowait``, which directly
   satisfies A's ``get_task`` future. The conn is bound to ``get_task``'s
   result, not in the queue.
3. Task C calls ``pool.close()``. Drain runs and the queue is empty
   (the conn went to A's get_task). ``_drain_complete = True``.
4. Task A receives an outer cancel (TaskGroup parent / asyncio.timeout).
   Enters the ``except BaseException`` arm. ``get_task.done()`` is True;
   calls ``_put_back_or_release_late_winner(get_task.result())``.
5. Without the closed re-check, ``put_nowait`` succeeds. The conn lands
   in the closed pool's queue; ``_drain_complete`` was already True, so
   no one drains again. ``ResourceWarning: unclosed socket`` at GC.

Mirrors the analogous closed-pool re-check in ``_release`` —
``put_nowait`` into a closed queue parks the conn for no one to
drain.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_put_back_late_winner_into_closed_pool_closes_conn_and_releases_reservation() -> None:
    """Directly drive ``_put_back_or_release_late_winner`` with
    ``self._closed = True``: the helper must close the conn AND
    release the reservation; the queue must stay empty.
    """
    # Construct a pool stub bypassing the public initialize() path.
    pool = ConnectionPool.__new__(ConnectionPool)
    import asyncio
    import threading

    pool._closed = True
    pool._pool = asyncio.Queue()
    pool._size = 1
    pool._lock = asyncio.Lock()
    # Stubs required by _release_reservation; not strictly all used.
    pool._closed_event = asyncio.Event()
    pool._closed_event.set()
    pool._drain_complete = True
    pool._max_size = 2

    # _release_reservation signals; stub the state-change wake.
    pool._signal_state_change = MagicMock()

    conn = MagicMock()
    conn.close = AsyncMock()
    conn._lock_owner = threading.get_ident()

    await pool._put_back_or_release_late_winner(conn)

    # Conn was closed (not put back into a closed pool's queue).
    conn.close.assert_awaited_once()
    # Queue stayed empty.
    assert pool._pool.empty()
    # Reservation slot was released.
    assert pool._size == 0


@pytest.mark.asyncio
async def test_put_back_late_winner_into_open_pool_returns_conn_to_queue() -> None:
    """Negative twin: on an OPEN pool, the helper still puts the
    conn back into the queue (the pre-fix happy path)."""
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

    # Conn was put back into the queue, NOT closed.
    conn.close.assert_not_awaited()
    assert pool._pool.qsize() == 1
    assert pool._size == 1
