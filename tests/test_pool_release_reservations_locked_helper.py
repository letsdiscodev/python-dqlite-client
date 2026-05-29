"""Pin: bulk reservation decrements route through the same underflow-guarded
helper (``_release_reservations_locked(n)``, caller holds the lock) as the per-conn
``_release_reservation`` path, so initialize's partial-warm-up decrement can no
longer drive _size negative past max_size.
"""

from __future__ import annotations

import logging

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_release_reservations_locked_normal_decrement() -> None:
    """Happy path: decrement under capacity returns True and reduces _size."""
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
    """Over-decrement is refused with the canonical ERROR log, keeping _size
    non-negative."""
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
async def test_release_reservations_locked_underflow_signals_state_change() -> None:
    """Pin: the underflow-refusal arm also calls _signal_state_change() so a parked
    acquirer re-evaluates against the current _size instead of waiting on a wake
    signal that the earlier double-release may have lost."""
    from unittest.mock import MagicMock

    pool = ConnectionPool(["localhost:9001"])
    spy = MagicMock(wraps=pool._signal_state_change)
    pool._signal_state_change = spy

    async with pool._lock:
        pool._size = 2
        ok = pool._release_reservations_locked(5)
    assert ok is False
    spy.assert_called_once()


@pytest.mark.asyncio
async def test_release_reservations_locked_rejects_zero_and_negative_n() -> None:
    """The helper requires n >= 1: n=0 spuriously wakes every acquirer, and n<0
    would slip past the guard and INCREMENT _size. Reject with ValueError."""
    pool = ConnectionPool(["localhost:9001"])
    async with pool._lock:
        pool._size = 5
        for bad_n in (0, -1, -100):
            with pytest.raises(ValueError, match="n >= 1"):
                pool._release_reservations_locked(bad_n)
        assert pool._size == 5


@pytest.mark.asyncio
async def test_release_reservations_locked_rejects_bool_and_non_int() -> None:
    """isinstance(True, int) is True so True would decrement by 1; reject bool
    and other non-int types explicitly."""
    pool = ConnectionPool(["localhost:9001"])
    async with pool._lock:
        pool._size = 5
        for bad_n in (True, False, 1.5, "1", None):
            with pytest.raises(ValueError, match="n >= 1"):
                pool._release_reservations_locked(bad_n)  # type: ignore[arg-type]
        assert pool._size == 5


@pytest.mark.asyncio
async def test_release_reservations_locked_requires_lock_held() -> None:
    """Pin the runtime assertion that the caller holds self._lock."""
    pool = ConnectionPool(["localhost:9001"])
    pool._size = 5
    # NOT inside `async with pool._lock`.
    with pytest.raises(AssertionError):
        pool._release_reservations_locked(1)


@pytest.mark.asyncio
async def test_release_reservation_at_zero_still_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_release_reservation() (n=1) underflow log still fires through the
    shared helper."""
    pool = ConnectionPool(["localhost:9001"])
    assert pool._size == 0

    with caplog.at_level(logging.ERROR, logger="dqliteclient.pool"):
        await pool._release_reservation()

    assert pool._size == 0
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err_records) == 1
    assert "_release_reservations_locked" in err_records[0].message
