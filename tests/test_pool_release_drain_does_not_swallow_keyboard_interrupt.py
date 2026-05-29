"""Pin: ``_release``'s shielded ``_pending_drain`` await suppresses CancelledError
and Exception but MUST NOT suppress arbitrary BaseException (e.g. KeyboardInterrupt /
SystemExit) — Ctrl+C must escape the cleanup path.

Uses a custom BaseException subclass so pytest's top-level handling does not
intercept the signal.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.pool import ConnectionPool


class _SyntheticControlPlaneSignal(BaseException):
    pass


@pytest.mark.asyncio
async def test_drain_base_exception_propagates_through_release() -> None:
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    pool._size = 1
    conn = DqliteConnection("localhost:9001")
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    conn._in_transaction = True

    async def signaling_drain() -> None:
        await asyncio.sleep(0)
        raise _SyntheticControlPlaneSignal("control-plane signal")

    conn._pending_drain = asyncio.create_task(signaling_drain())

    async def fake_reset(c: DqliteConnection) -> bool:
        return False

    pool._reset_connection = fake_reset  # type: ignore[assignment]

    async def noop() -> None:
        return

    pool._release_reservation = noop

    async def fake_close() -> None:
        return

    conn.close = fake_close

    with pytest.raises(_SyntheticControlPlaneSignal):
        await pool._release(conn)
