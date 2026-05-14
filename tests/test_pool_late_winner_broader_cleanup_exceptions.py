"""Pin: ``_put_back_or_release_late_winner``'s two close arms suppress
the canonical ``_POOL_CLEANUP_EXCEPTIONS`` tuple, not just ``OSError``.

The sibling ``_release`` path at ``pool.py:1626-1661`` catches
``_POOL_CLEANUP_EXCEPTIONS`` (``OSError + DqliteConnectionError +
ProtocolError + OperationalError + InterfaceError``) so a late-winner
conn whose ``close()`` raises any pool-cleanup-class exception
doesn't escape the suppression. Pre-fix, only ``OSError`` was
suppressed: a ``DqliteConnectionError`` or ``InterfaceError`` from a
half-torn-down transport escaped, the ``finally`` restored the flag,
and ``_release_reservation`` was NEVER reached — the slot leaked
permanently for the lifetime of the pool.

Both the closed-pool arm and the QueueFull arm carry the same
discipline; pin both.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import (
    DqliteConnectionError,
    InterfaceError,
)
from dqliteclient.pool import ConnectionPool


def _make_conn_close_raises(exc: BaseException) -> MagicMock:
    conn = MagicMock()
    conn._pool_released = True

    async def _close() -> None:
        # Emulate the real ``DqliteConnection.close`` early-return at
        # connection.py:1621-1622: pool-released conns short-circuit.
        if conn._pool_released:
            return
        raise exc

    conn.close = AsyncMock(side_effect=_close)
    return conn


@pytest.mark.asyncio
async def test_closed_pool_arm_suppresses_dqlite_connection_error() -> None:
    """A late-winner conn whose ``close()`` raises
    ``DqliteConnectionError`` (e.g. peer reset between checkout and
    close) must NOT leak the slot. The reservation must be released
    so the pool counter stays consistent."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1

    conn = _make_conn_close_raises(DqliteConnectionError("peer reset"))
    # Must not raise.
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 0, (
        f"closed-pool arm leaked the reservation slot when close() "
        f"raised DqliteConnectionError; _size={pool._size} (want 0)"
    )


@pytest.mark.asyncio
async def test_closed_pool_arm_suppresses_interface_error() -> None:
    """Same shape, ``InterfaceError`` from the cross-loop / tx-owner
    guard — must be in the suppressed tuple."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=2, timeout=1.0)
    pool._closed = True
    pool._size = 1

    conn = _make_conn_close_raises(InterfaceError("owned by another task"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 0, (
        f"closed-pool arm leaked the reservation slot when close() "
        f"raised InterfaceError; _size={pool._size} (want 0)"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_suppresses_dqlite_connection_error() -> None:
    """The QueueFull arm shares the same close-and-release discipline
    and must also suppress the canonical cleanup tuple."""
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    pool._pool.put_nowait(MagicMock())
    pool._size = 2

    conn = _make_conn_close_raises(DqliteConnectionError("peer reset"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 1, (
        f"QueueFull arm leaked the reservation slot when close() "
        f"raised DqliteConnectionError; _size={pool._size} (want 1)"
    )


@pytest.mark.asyncio
async def test_queue_full_arm_suppresses_interface_error() -> None:
    pool = ConnectionPool(["127.0.0.1:9001"], max_size=1, timeout=1.0)
    pool._pool.put_nowait(MagicMock())
    pool._size = 2

    conn = _make_conn_close_raises(InterfaceError("owned by another task"))
    await pool._put_back_or_release_late_winner(conn)
    assert pool._size == 1, (
        f"QueueFull arm leaked the reservation slot when close() "
        f"raised InterfaceError; _size={pool._size} (want 1)"
    )
