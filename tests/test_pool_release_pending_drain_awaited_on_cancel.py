"""Pin: ``_release``'s finally drains ``_pending_drain`` before setting
``_pool_released=True``. Otherwise close()'s pending-drain block is short-circuited
by the flag, the reader-task outlives the connection, and shutdown logs
"Task was destroyed but it is pending" under cancel-heavy workloads.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_release_drains_pending_before_setting_pool_released() -> None:
    """Pin: the drain task completes BEFORE ``_pool_released`` flips.

    _release_reservation runs after the flag is set, so its stub capturing
    drain.done() tells us whether the pre-flag drain ran.
    """
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1  # reservation already taken
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True  # force the ROLLBACK branch in _reset_connection

    # Multi-turn drain so a single _release_reservation yield can't drain it;
    # only the explicit shield(pending) await can, before _pool_released flips.
    async def slow_drain() -> None:
        for _ in range(5):
            await asyncio.sleep(0)

    conn._pending_drain = asyncio.create_task(slow_drain())

    async def fake_reset(c: DqliteConnection) -> bool:
        return False  # close-and-drop branch (mimics cancel-during-ROLLBACK)

    pool._reset_connection = fake_reset  # type: ignore[assignment]

    drain_done_when_release_reservation_ran: list[bool] = []

    async def observing_release_reservation() -> None:
        drain_done_when_release_reservation_ran.append(
            conn._pending_drain is not None and conn._pending_drain.done()
        )

    pool._release_reservation = observing_release_reservation

    async def fake_close() -> None:
        return

    with patch.object(conn, "close", new=fake_close):
        await pool._release(conn)

    assert drain_done_when_release_reservation_ran == [True], (
        "drain task must have completed before _release_reservation ran "
        "(i.e. before _pool_released was set in the finally)"
    )
    assert conn._pool_released is True
    assert conn._pending_drain.done()


@pytest.mark.asyncio
async def test_release_with_no_pending_drain_path_unchanged() -> None:
    """Negative pin: with no _pending_drain, the drain block is a no-op."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True
    assert conn._pending_drain is None

    async def fake_reset(c: DqliteConnection) -> bool:
        return False

    pool._reset_connection = fake_reset  # type: ignore[assignment]

    async def noop() -> None:
        return

    pool._release_reservation = noop

    async def fake_close() -> None:
        return

    with patch.object(conn, "close", new=fake_close):
        await pool._release(conn)

    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_release_pending_drain_failure_does_not_block_release() -> None:
    """If the drain task raises, _release still completes (sets _pool_released,
    releases the reservation); the drain's exception is observed and discarded."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True

    async def failing_drain() -> None:
        raise RuntimeError("drain failed")

    conn._pending_drain = asyncio.create_task(failing_drain())

    async def fake_reset(c: DqliteConnection) -> bool:
        return False

    pool._reset_connection = fake_reset  # type: ignore[assignment]

    async def noop() -> None:
        return

    pool._release_reservation = noop

    async def fake_close() -> None:
        return

    with patch.object(conn, "close", new=fake_close):
        await pool._release(conn)

    assert conn._pool_released is True
    assert conn._pending_drain.done()
