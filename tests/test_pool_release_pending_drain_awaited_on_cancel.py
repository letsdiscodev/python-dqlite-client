"""Pin: ``_release``'s finally drains ``_pending_drain`` before
setting ``_pool_released=True``.

Cancel mid-ROLLBACK (during ``_release._reset_connection``) lands in
``_run_protocol``'s cancellation branch, which schedules a bounded
``wait_closed`` drain task on ``_pending_drain``. ``close()`` would
normally await that drain at its pending-drain block, but the
``_pool_released=True`` flag set by ``_release``'s finally short-
circuits past it. The reader-task then outlives the connection,
producing "Task was destroyed but it is pending" warnings on
shutdown under cancel-heavy workloads.

The fix snapshots and shield-awaits the drain BEFORE setting
``_pool_released=True``.
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

    Use a release_reservation stub that captures the drain's done-state
    at the moment it runs. ``_release_reservation`` is awaited AFTER
    ``_pool_released = True`` in the finally clause, so capturing
    ``drain.done()`` from inside the stub tells us whether the fix's
    pre-flag drain ran.
    """
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1  # reservation already taken
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True  # force the ROLLBACK branch in _reset_connection

    # Drain task that requires multiple loop turns to complete — so a
    # single yield from _release_reservation cannot accidentally drain
    # it. The fix's explicit ``await asyncio.shield(pending)`` is the
    # only mechanism that drains it before _pool_released flips.
    async def slow_drain() -> None:
        for _ in range(5):
            await asyncio.sleep(0)

    conn._pending_drain = asyncio.create_task(slow_drain())

    async def fake_reset(c: DqliteConnection) -> bool:
        # Take the close-and-drop branch (mimics
        # cancel-during-ROLLBACK). The pending drain is what we are
        # pinning the order on.
        return False

    pool._reset_connection = fake_reset  # type: ignore[method-assign]

    # Capture the drain task's done-state at the moment
    # _release_reservation runs. With the fix, the drain has been
    # explicitly awaited before this point. Without the fix, the
    # drain is still pending when this fires.
    drain_done_when_release_reservation_ran: list[bool] = []

    async def observing_release_reservation() -> None:
        drain_done_when_release_reservation_ran.append(
            conn._pending_drain is not None and conn._pending_drain.done()
        )

    pool._release_reservation = observing_release_reservation  # type: ignore[method-assign]

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
    """Negative pin: when there is no ``_pending_drain``, the existing
    behaviour is unchanged — the drain block is a no-op."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True
    assert conn._pending_drain is None

    async def fake_reset(c: DqliteConnection) -> bool:
        return False

    pool._reset_connection = fake_reset  # type: ignore[method-assign]

    async def noop() -> None:
        return

    pool._release_reservation = noop  # type: ignore[method-assign]

    async def fake_close() -> None:
        return

    with patch.object(conn, "close", new=fake_close):
        await pool._release(conn)

    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_release_pending_drain_failure_does_not_block_release() -> None:
    """If the drain task itself raises, ``_release`` must still
    complete (set ``_pool_released``, release the reservation). The
    fix uses ``contextlib.suppress(BaseException)`` around the
    shielded await so the drain's exception is observed and discarded
    — ``_release`` is a cleanup path; the original cause is what the
    user cares about."""
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

    pool._reset_connection = fake_reset  # type: ignore[method-assign]

    async def noop() -> None:
        return

    pool._release_reservation = noop  # type: ignore[method-assign]

    async def fake_close() -> None:
        return

    with patch.object(conn, "close", new=fake_close):
        await pool._release(conn)

    assert conn._pool_released is True
    # The drain task is done — the suppression observed the exception.
    assert conn._pending_drain.done()
