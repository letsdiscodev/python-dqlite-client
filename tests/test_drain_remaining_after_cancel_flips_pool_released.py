"""_drain_remaining_after_cancel clears conn._pool_released before close() so the close runs.

Queued conns carry _pool_released=True (set by _release), and DqliteConnection.close()
short-circuits on that guard; without the flip the drain leaks one transport + task per conn.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _make_conn_emulating_release_flag() -> MagicMock:
    """Connection mock emulating DqliteConnection.close's early-return on _pool_released."""
    conn = MagicMock()
    conn._pool_released = True
    close_actually_ran: list[bool] = []

    async def _close() -> None:
        if conn._pool_released:
            close_actually_ran.append(False)
            return
        close_actually_ran.append(True)

    conn.close = AsyncMock(side_effect=_close)
    conn._close_events = close_actually_ran
    return conn


@pytest.mark.asyncio
async def test_drain_remaining_after_cancel_clears_flag_before_close() -> None:
    """Each queued conn's close() must tear down the transport, not short-circuit on the guard."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=3, timeout=1.0)

    conn_a = _make_conn_emulating_release_flag()
    conn_b = _make_conn_emulating_release_flag()
    conn_c = _make_conn_emulating_release_flag()
    pool._pool.put_nowait(conn_a)
    pool._pool.put_nowait(conn_b)
    pool._pool.put_nowait(conn_c)
    pool._size = 3

    await pool._drain_remaining_after_cancel()

    # Each conn's close() saw _pool_released=False at entry, i.e. teardown ran.
    for conn in (conn_a, conn_b, conn_c):
        assert any(conn._close_events), (
            "drain_remaining_after_cancel must clear conn._pool_released "
            "before close() so the close actually tears down the "
            "transport; the early-return on _pool_released at the top "
            "of DqliteConnection.close otherwise turns close() into a "
            "no-op and the transport / reader task leak"
        )
    assert pool._size == 0, f"reservation accounting drifted: _size={pool._size}, want 0"
