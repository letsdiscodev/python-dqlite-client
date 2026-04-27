"""Pin the underflow guard on ``ConnectionPool._release_reservation``.

Every reservation slot in the pool corresponds to a prior
``self._size += 1`` and the only decrement path is through
``_release_reservation`` — so ``_size <= 0`` here is unreachable
under correct accounting. The guard is defensive: a future refactor
that double-decrements (e.g., the cancel-shielded paths landing two
``_release_reservation()`` calls per slot) would otherwise silently
produce a negative ``_size`` that passes every
``self._size < self._max_size`` capacity check, expanding the pool
beyond its stated bound and surfacing only as "the pool is way
bigger than max_size" reports from operators.

These tests pin:

* The decrement is refused — ``_size`` stays at zero rather than
  going negative.
* An ERROR-level log record is emitted so the bug is observable.
* Repeated calls remain safe (idempotent at zero).
"""

from __future__ import annotations

import logging

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_release_reservation_at_zero_does_not_underflow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Calling ``_release_reservation`` on a pool with ``_size == 0``
    must NOT decrement to ``-1``. The guard logs at ERROR and
    returns early."""
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
    """Multiple bogus releases at zero stay at zero — accounting
    invariants hold even if the bug-source repeats."""
    pool = ConnectionPool(["localhost:9001"])

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()
        await pool._release_reservation()
        await pool._release_reservation()

    assert pool._size == 0
    # One ERROR per call — operators see every double-release.
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err_records) == 3


@pytest.mark.asyncio
async def test_release_reservation_after_grant_decrements_normally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative pin: the guard only fires on underflow. A normal
    (paired) increment-then-decrement still works without an ERROR
    log."""
    pool = ConnectionPool(["localhost:9001"])
    # Simulate an acquire's reservation grant: increment ``_size``
    # under the lock just like ``acquire`` does at the grant site.
    async with pool._lock:
        pool._size += 1
    assert pool._size == 1

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()

    assert pool._size == 0
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert err_records == [], "no ERROR record on a paired release"
