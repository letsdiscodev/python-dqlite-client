"""Pin: ``_drain_remaining_after_cancel`` bounds each connection's
close attempt by ``close_timeout + 0.5`` so one hung connection
does not block the rest of the queue from being cleaned up.

The per-iteration ``asyncio.wait_for(...)`` is the upper bound that
holds even when the inner ``close_timeout`` discipline has a bug.
The ``+0.5`` headroom absorbs the graceful-vs-force step inside
``_close_impl`` without truncating it. The ``except TimeoutError``
arm logs at WARNING with the ``"abandoning to drain"`` substring;
operators grep for that marker when debugging stuck shutdowns.

A regression that drops the ``wait_for`` would re-introduce the
unbounded-block hazard. A regression that re-raises ``TimeoutError``
would abort the drain loop on the first hung conn and leak the rest.
A regression that swallows ``TimeoutError`` silently would lose the
operator-visible audit trail.
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
    """A hung close() is abandoned with a WARN log; the next queued
    connection still gets its close attempt."""
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

    # Stub the release-reservation to avoid touching the lock /
    # signal infrastructure beyond the unit under test.
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


def test_drain_remaining_after_cancel_source_carries_per_iteration_timeout() -> None:
    """Inspection pin: the ``asyncio.wait_for`` envelope must remain
    in the source and the timeout must be derived from
    ``_DRAIN_PER_CONN_CAP_MULTIPLIER`` rather than a bare ``+ 0.5``
    literal. A regression that drops the wait_for, reverts to the
    magic literal, or short-circuits the cap would re-introduce
    either the unbounded-block hazard or the truncate-graceful-
    drain hazard the cap was sized to prevent."""
    import inspect

    src = inspect.getsource(ConnectionPool._drain_remaining_after_cancel)
    assert "asyncio.wait_for" in src
    assert "_DRAIN_PER_CONN_CAP_MULTIPLIER" in src
    assert "self._close_timeout + 0.5" not in src, (
        "magic ``+ 0.5`` literal must not return — the cap is derived "
        "from ``_CLOSE_RESNAPSHOT_CAP + 1`` via ``_DRAIN_PER_CONN_CAP_MULTIPLIER``"
    )
    assert "except TimeoutError" in src
    assert "abandoning to drain" in src
