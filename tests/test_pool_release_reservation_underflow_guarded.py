"""Pin the underflow guard on ``ConnectionPool._release_reservation``.

A double-decrement would otherwise drive _size negative, passing every
_size < _max_size capacity check and expanding the pool past its bound.
"""

from __future__ import annotations

import logging

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_release_reservation_at_zero_does_not_underflow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_release_reservation at _size == 0 does not decrement to -1; logs ERROR."""
    pool = ConnectionPool(["localhost:9001"])
    assert pool._size == 0

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()

    assert pool._size == 0
    assert any(
        "_release_reservation" in record.message and record.levelno == logging.ERROR
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_release_reservation_at_zero_is_idempotent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Multiple bogus releases at zero stay at zero."""
    pool = ConnectionPool(["localhost:9001"])

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()
        await pool._release_reservation()
        await pool._release_reservation()

    assert pool._size == 0
    # One ERROR per call.
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err_records) == 3


@pytest.mark.asyncio
async def test_release_reservation_after_grant_decrements_normally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative pin: a paired increment-then-decrement works without an ERROR log."""
    pool = ConnectionPool(["localhost:9001"])
    # Simulate an acquire's reservation grant.
    async with pool._lock:
        pool._size += 1
    assert pool._size == 1

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()

    assert pool._size == 0
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert err_records == [], "no ERROR record on a paired release"
