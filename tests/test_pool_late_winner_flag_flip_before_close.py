"""``_put_back_or_release_late_winner`` must clear ``conn._pool_released``
before ``conn.close()`` on both arms; otherwise close() early-returns on the
``_pool_released`` guard, leaking the reader task and transport while
``_release_reservation`` still decrements ``_size``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _make_conn_with_released_flag() -> MagicMock:
    """Mock emulating DqliteConnection.close()'s early-return on ``_pool_released``."""
    conn = MagicMock()
    conn._pool_released = True
    close_called: list[bool] = []

    async def _close() -> None:
        # Mirror DqliteConnection.close: pool-released conns short-circuit.
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
    pool._size = 1

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)

    assert any(conn._close_events), (
        "closed-pool arm must clear conn._pool_released before close() so "
        "the close actually runs; otherwise the connection leaks"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_flips_flag_so_close_actually_runs() -> None:
    # max_size=1 so the queue fills and put_nowait raises QueueFull.
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    pool._pool.put_nowait(MagicMock())
    pool._size = 2

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)

    assert any(conn._close_events), (
        "QueueFull arm must clear conn._pool_released before close() so "
        "the close actually runs; otherwise the connection leaks"
    )


@pytest.mark.asyncio
async def test_closed_pool_arm_does_not_leave_flag_false() -> None:
    """The flag is restored to True in the finally so a stale-reference second
    close() falls through the early-return rather than re-running on a dead conn."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)
    assert conn._pool_released is True, (
        "the flag must be restored to True after the close so the "
        "documented stale-reference early-return discipline survives"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_does_not_leave_flag_false() -> None:
    """The QueueFull arm must also restore ``conn._pool_released`` to True in its
    finally so a later stale-reference second close() falls through the early-return."""
    # max_size=1 so the queue fills and put_nowait raises QueueFull (open-pool arm).
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    pool._pool.put_nowait(MagicMock())
    pool._size = 2
    pool._closed = False  # pool open — this is the QueueFull case

    conn = _make_conn_with_released_flag()
    await pool._put_back_or_release_late_winner(conn)
    assert conn._pool_released is True, (
        "QueueFull arm must restore the flag to True after close so "
        "the documented stale-reference early-return discipline survives"
    )
