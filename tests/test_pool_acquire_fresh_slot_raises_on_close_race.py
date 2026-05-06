"""``pool.acquire()`` must not yield a fresh connection from the
fresh-slot reservation arm on a closed pool.

The fresh-slot arm reserves a new ``_size`` slot under ``_lock`` then
drops the lock and awaits ``_create_connection`` (TCP handshake +
leader discovery). ``close()`` racing this await flips
``self._closed = True`` while the create is in-flight; without a
post-create ``_closed`` re-check, the fresh connection is yielded on
a pool whose flag is True — contract violation, sneaky leak (user
runs real queries against an invisibly-closed pool until they exit
the ``async with`` block; ``_release`` then closes the conn but the
queries already ran).

The dead-conn-replacement arm at ``pool.py:1202-1210`` already has
this re-check; the fresh-slot arm did not.

Pin: a fresh-slot acquire racing close() must raise
``DqliteConnectionError("Pool is closed")`` AND close the freshly-built
connection (no transport leak).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_fresh_slot_raises_on_close_race() -> None:
    """close() landing between the fresh-slot _create_connection
    suspension and the yield must raise DqliteConnectionError and
    close the fresh connection — symmetric with the dead-conn-revive
    arm's discipline.
    """
    pool = ConnectionPool(["localhost:19001"], min_size=0, max_size=1, timeout=2.0)

    # Empty queue, fresh-slot arm taken (size=0, max_size=1).
    fresh_conn = MagicMock()
    fresh_conn.is_connected = True
    fresh_conn.close = AsyncMock()
    fresh_conn._address = "localhost:19001"
    fresh_conn._pool_released = False

    create_gate = asyncio.Event()

    async def _create() -> object:
        # Block until the test fires close(); release the fresh conn
        # only after close() has set self._closed = True and returned.
        await create_gate.wait()
        return fresh_conn

    pool._create_connection = _create  # type: ignore[assignment]

    async def _consume() -> None:
        async with pool.acquire() as conn:
            pytest.fail(f"acquire yielded on a closed pool: {conn!r}")

    consume_task = asyncio.create_task(_consume())

    # Wait for the fresh-slot arm to reach _create_connection.
    for _ in range(50):
        await asyncio.sleep(0)
        if create_gate._waiters:  # consumer is parked on create_gate.wait()
            break

    # Race close() past the suspended _create_connection.
    await pool.close()
    # Now let the fresh-slot arm resume and produce the fresh conn.
    create_gate.set()

    with pytest.raises(DqliteConnectionError, match="Pool is closed"):
        await asyncio.wait_for(consume_task, timeout=1.0)

    # The fresh conn must have been closed (no transport leak).
    fresh_conn.close.assert_awaited()
    # The reservation slot must have been released back to the pool —
    # otherwise the pool wedges at ``_size = max_size`` even though
    # nothing is checked out. Catches a regression where the
    # ``_release_reservation()`` shield is dropped from the new arm
    # (the close-awaited assertion alone does NOT catch this).
    assert pool._size == 0, f"reservation slot leaked on close-race: pool._size={pool._size}"
