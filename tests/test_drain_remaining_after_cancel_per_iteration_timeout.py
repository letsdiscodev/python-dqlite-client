"""_drain_remaining_after_cancel bounds each close by a per-iteration wait_for so one hung conn
does not block the rest of the queue. The except TimeoutError arm logs WARNING with the
"abandoning to drain" marker that operators grep for when debugging stuck shutdowns.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_drain_remaining_after_cancel_per_iteration_timeout_abandons_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hung close() is abandoned with a WARN log; the next queued conn still gets its attempt."""
    pool = ConnectionPool(["a:9001", "b:9001"], min_size=0, max_size=2)
    pool._close_timeout = 0.05

    hung_called = asyncio.Event()
    fast_called = asyncio.Event()

    async def _hang_forever() -> None:
        hung_called.set()
        await asyncio.sleep(60)

    async def _fast_close() -> None:
        fast_called.set()

    hung_conn = MagicMock()
    hung_conn._address = "hung:9001"
    hung_conn._pool_released = True
    hung_conn.close = _hang_forever

    fast_conn = MagicMock()
    fast_conn._address = "fast:9001"
    fast_conn._pool_released = True
    fast_conn.close = _fast_close

    pool._pool.put_nowait(hung_conn)
    pool._pool.put_nowait(fast_conn)

    # Stub release-reservation to avoid touching the lock/signal infrastructure.
    pool._release_reservation = MagicMock(
        side_effect=lambda: asyncio.sleep(0),
    )

    with caplog.at_level(logging.WARNING, logger="dqliteclient.pool"):
        await pool._drain_remaining_after_cancel()

    assert hung_called.is_set(), "hung connection's close was attempted"
    assert fast_called.is_set(), "the loop continued to the next queued conn"

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "abandoning to drain" in r.message
    ]
    assert warnings, (
        f"expected WARN log naming the hung connection; got: {[r.message for r in caplog.records]}"
    )
