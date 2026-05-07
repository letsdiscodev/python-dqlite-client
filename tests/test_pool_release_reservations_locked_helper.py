"""Pin: bulk reservation decrements in ``ConnectionPool`` route
through the same under-flow-guarded helper as the per-conn
``_release_reservation`` path.

``initialize`` recovers from partial warm-up by decrementing
``_size`` by ``unqueued`` (the count of reservations that never
made it into the queue). The decrement historically bypassed the
``_release_reservation`` helper and its under-flow guard, so a
future double-decrement at this site would silently produce a
negative ``_size`` and the pool would expand past ``max_size``.

The fix extracts the under-flow-guarded decrement into
``_release_reservations_locked(n)`` (caller holds the lock),
shared by both the per-conn helper and the bulk-decrement path. A
deliberate over-decrement now lands the canonical ERROR log AND
refuses to underflow ``_size``.
"""

from __future__ import annotations

import logging

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_release_reservations_locked_normal_decrement() -> None:
    """Happy path: decrement under capacity returns True and reduces
    ``_size``."""
    pool = ConnectionPool(["localhost:9001"])
    async with pool._lock:
        pool._size = 5
        ok = pool._release_reservations_locked(3)
    assert ok is True
    assert pool._size == 2


@pytest.mark.asyncio
async def test_release_reservations_locked_underflow_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Over-decrement is refused with the canonical ERROR log,
    keeping ``_size`` non-negative. The bulk-decrement site (and any
    future site) inherits this guard."""
    pool = ConnectionPool(["localhost:9001"])
    async with pool._lock:
        pool._size = 2
        with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
            ok = pool._release_reservations_locked(5)
    assert ok is False
    assert pool._size == 2  # unchanged
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err_records) == 1
    assert "_release_reservations_locked" in err_records[0].message
    assert "_size=2" in err_records[0].message
    assert "n=5" in err_records[0].message


@pytest.mark.asyncio
async def test_release_reservation_at_zero_still_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Existing ``_release_reservation()`` (n=1) under-flow log
    still fires through the shared helper. Pin against a regression
    that loses the diagnostic when consolidating decrement paths."""
    pool = ConnectionPool(["localhost:9001"])
    assert pool._size == 0

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()

    assert pool._size == 0
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err_records) == 1
    assert "_release_reservations_locked" in err_records[0].message
