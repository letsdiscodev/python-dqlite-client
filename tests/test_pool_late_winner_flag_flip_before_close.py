"""``ConnectionPool._put_back_or_release_late_winner`` must flip
``conn._pool_released`` to ``False`` BEFORE calling ``conn.close()`` on
both the closed-pool short-circuit and the QueueFull arm.

The connection enters this method with ``_pool_released = True`` (set
by the prior ``_release`` write before the put_nowait). Without the
flip, ``DqliteConnection.close()`` early-returns at the
``if self._pool_released: return`` guard (connection.py:1621-1622) —
the close becomes a no-op, the reader task and transport leak, and
``_release_reservation`` decrements ``_size`` as though the close had
happened. Net: one fully-resourced connection leaks per occurrence.

Sibling pool-owned close site ``_drain_idle`` (pool.py:1069-1118)
brackets every close in ``_pool_released = False`` /
``= True`` discipline. This file pins the same discipline on the
``_put_back_or_release_late_winner`` two arms.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _make_conn_with_released_flag() -> MagicMock:
    """A connection mock that emulates DqliteConnection.close()'s
    early-return on ``_pool_released``."""
    conn = MagicMock()
    conn._pool_released = True
    close_called: list[bool] = []

    async def _close() -> None:
        # Emulate the real ``DqliteConnection.close`` early-return at
        # connection.py:1621-1622: pool-released conns short-circuit.
        if conn._pool_released:
            close_called.append(False)
            return
        close_called.append(True)

    conn.close = AsyncMock(side_effect=_close)
    conn._close_events = close_called
    return conn


@pytest.mark.asyncio
async def test_closed_pool_arm_flips_flag_so_close_actually_runs() -> None:
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1  # one outstanding reservation to release

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)

    # Without the flag flip, conn.close() returns early and never
    # tears down the transport.
    assert any(conn._close_events), (
        "closed-pool arm must clear conn._pool_released before close() so "
        "the close actually runs; otherwise the connection leaks "
        "(connection.py:1621-1622 early-returns when _pool_released=True)"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_flips_flag_so_close_actually_runs() -> None:
    # max_size=1 so we can fill the underlying _pool to capacity and
    # force put_nowait to raise QueueFull.
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    # Fill the queue with a placeholder so put_nowait raises QueueFull.
    pool._pool.put_nowait(MagicMock())
    pool._size = 2  # reservation accounting we'll decrement back to 1

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)

    assert any(conn._close_events), (
        "QueueFull arm must clear conn._pool_released before close() so "
        "the close actually runs; otherwise the connection leaks"
    )


@pytest.mark.asyncio
async def test_closed_pool_arm_does_not_leave_flag_false() -> None:
    """Contract symmetry with the _drain_idle pattern: the flag is
    restored to True in the finally so any stale-reference second
    close() call from somewhere else falls through the documented
    early-return rather than re-running close on a dead conn."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)
    assert conn._pool_released is True, (
        "the flag must be restored to True after the close so the "
        "documented stale-reference early-return discipline survives"
    )
