"""ConnectionPool emits ResourceWarning when GC'd without await close(),
but only once a slot has been reserved (avoid false positives on an
unused pool).
"""

import asyncio
import warnings
from typing import Any

import pytest

from dqliteclient.pool import _pool_unclosed_warning


def test_unclosed_warning_skips_when_never_reserved() -> None:
    closed_flag = [False]
    reserved_flag = [False]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag)
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_skips_when_close_was_called() -> None:
    closed_flag = [True]
    reserved_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag)
    assert not any(issubclass(rec.category, ResourceWarning) for rec in w)


def test_unclosed_warning_fires_when_reserved_but_not_closed() -> None:
    closed_flag = [False]
    reserved_flag = [True]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag)
    matching = [r for r in w if issubclass(r.category, ResourceWarning)]
    assert len(matching) == 1
    assert "ConnectionPool" in str(matching[0].message)


def test_finalizer_registered_on_construction() -> None:
    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    assert pool._finalizer is not None
    assert pool._finalizer.alive
    pool._finalizer.detach()


@pytest.mark.asyncio
async def test_unclosed_warning_omits_count_when_queue_empty() -> None:
    """Empty queue keeps the short message (no "0 queued connection(s)" clause)."""
    closed_flag = [False]
    reserved_flag = [True]
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag, queue)
    matching = [r for r in w if issubclass(r.category, ResourceWarning)]
    assert len(matching) == 1
    msg = str(matching[0].message)
    assert "ConnectionPool was garbage-collected" in msg
    assert "queued connection(s) will each emit" not in msg


@pytest.mark.asyncio
async def test_unclosed_warning_reports_queued_count_when_queue_nonempty() -> None:
    """Queued connections present: message names the count so the operator
    can correlate the N+1 warnings to one root cause."""
    closed_flag = [False]
    reserved_flag = [True]
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
    # Sentinels suffice; qsize only depends on the put count.
    for _ in range(5):
        queue.put_nowait(object())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _pool_unclosed_warning(closed_flag, reserved_flag, queue)
    matching = [r for r in w if issubclass(r.category, ResourceWarning)]
    assert len(matching) == 1
    msg = str(matching[0].message)
    assert "5 queued connection(s)" in msg
    assert "same root cause" in msg


def test_finalizer_arg_tuple_carries_queue() -> None:
    """Finalizer registration must include the queue so it can read qsize at warn time."""
    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool(addresses=["h:9001"], min_size=0, max_size=2)
    finalizer = pool._finalizer
    assert finalizer is not None
    try:
        peeked = finalizer.peek()  # (obj, func, args, kwargs), or None if fired
        assert peeked is not None
        _obj, _func, args, _kwargs = peeked
        assert pool._pool in args, (
            "pool finalizer registration dropped the queue arg — "
            "queued-count message clause cannot fire"
        )
    finally:
        finalizer.detach()
