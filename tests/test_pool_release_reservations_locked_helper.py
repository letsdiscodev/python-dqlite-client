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
async def test_release_reservations_locked_underflow_signals_state_change() -> None:
    """Pin: the underflow-refusal arm also calls
    ``_signal_state_change()`` so any parked acquirer re-evaluates
    against the current ``_size`` rather than waiting for the next
    legitimate release. Underflow is symptomatic of an earlier double-
    release that may have lost a wake signal; the defensive wake
    bounds the worst-case acquirer wait when a double-release
    surfaces in production.

    Without the wake, an acquirer parked behind the lost signal
    waits indefinitely (or until another release lands), even though
    the current ``_size`` may be below capacity.
    """
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
    """The helper requires ``n >= 1``. ``n=0`` is a no-op decrement
    that would spuriously wake every parked acquirer; ``n<0`` would
    silently INCREMENT ``_size`` (since ``_size < n`` is False for
    non-negative ``_size`` against a negative ``n``, the under-flow
    guard fails open and ``_size -= -k`` increments).

    Reject explicitly with ``ValueError`` so future caller mistakes
    surface at the call site instead of corrupting capacity
    accounting silently.
    """
    pool = ConnectionPool(["localhost:9001"])
    async with pool._lock:
        pool._size = 5
        for bad_n in (0, -1, -100):
            with pytest.raises(ValueError, match="n >= 1"):
                pool._release_reservations_locked(bad_n)
        # _size is unchanged across every rejection.
        assert pool._size == 5


@pytest.mark.asyncio
async def test_release_reservations_locked_rejects_bool_and_non_int() -> None:
    """``isinstance(True, int)`` is True; ``True`` would decrement by
    1 silently. Reject ``bool`` and other non-int types explicitly."""
    pool = ConnectionPool(["localhost:9001"])
    async with pool._lock:
        pool._size = 5
        for bad_n in (True, False, 1.5, "1", None):
            with pytest.raises(ValueError, match="n >= 1"):
                pool._release_reservations_locked(bad_n)  # type: ignore[arg-type]
        assert pool._size == 5


@pytest.mark.asyncio
async def test_release_reservations_locked_requires_lock_held() -> None:
    """The docstring promises the caller already holds ``self._lock``.
    A future caller forgetting the lock corrupts ``_size`` under
    contention. Pin the runtime assertion."""
    pool = ConnectionPool(["localhost:9001"])
    pool._size = 5
    # NOT inside `async with pool._lock`.
    with pytest.raises(AssertionError):
        pool._release_reservations_locked(1)


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
