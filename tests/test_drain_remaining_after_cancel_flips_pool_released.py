"""Pin: ``_drain_remaining_after_cancel`` flips ``conn._pool_released``
to ``False`` before ``await conn.close()`` so the close actually runs.

Conns sitting in ``self._pool`` were placed there by ``_release`` and
carry ``_pool_released = True``. ``DqliteConnection.close()``
short-circuits at the ``if self._pool_released: return`` guard at the
top of the method — the close becomes a no-op. Without
the flip, the post-cancel drain leaks one transport + one reader
task per queued conn (the kernel times the socket out via
``CLOSE_WAIT``; the reader Task emits the "Task was destroyed but it
is pending" warning at interpreter shutdown).

Sibling sites (``_drain_idle`` at L1102-1121, the two
late-winner arms at L953/L980) all flip the flag for exactly this
reason. This file pins the missing flip on the
``_drain_remaining_after_cancel`` site.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _make_conn_emulating_release_flag() -> MagicMock:
    """A connection mock that emulates the real
    ``DqliteConnection.close`` early-return on ``_pool_released``."""
    conn = MagicMock()
    # Conns sitting in the queue carry _pool_released = True (set by
    # ``_release`` before put_nowait).
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
    """Each queued conn's ``close()`` must actually tear down the
    transport — not short-circuit at the ``_pool_released`` guard. The
    drain is pool-owned cleanup; the flag must come down for the
    close to pass the early-return contract."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=3, timeout=1.0)

    conn_a = _make_conn_emulating_release_flag()
    conn_b = _make_conn_emulating_release_flag()
    conn_c = _make_conn_emulating_release_flag()
    pool._pool.put_nowait(conn_a)
    pool._pool.put_nowait(conn_b)
    pool._pool.put_nowait(conn_c)
    pool._size = 3

    await pool._drain_remaining_after_cancel()

    # Each conn's close() saw _pool_released = False at entry, i.e.
    # the actual transport teardown ran.
    for conn in (conn_a, conn_b, conn_c):
        assert any(conn._close_events), (
            "drain_remaining_after_cancel must clear conn._pool_released "
            "before close() so the close actually tears down the "
            "transport; the early-return on _pool_released at the top "
            "of DqliteConnection.close otherwise turns close() into a "
            "no-op and the transport / reader task leak"
        )
    # Reservation accounting: each iteration decrements _size via
    # _release_reservation, so _size must be 0 after sweeping three.
    assert pool._size == 0, f"reservation accounting drifted: _size={pool._size}, want 0"
