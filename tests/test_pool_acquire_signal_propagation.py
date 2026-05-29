"""``ConnectionPool.acquire`` cleanup must not swallow KeyboardInterrupt /
SystemExit.

The cancel-and-drain-task sites in ``acquire()`` silence only
``CancelledError`` from the drained await; ``KeyboardInterrupt`` /
``SystemExit`` (and any other BaseException) must propagate per the signal
contract. The tests make the ``closed_task`` raise the chosen exception
on cancel and drive ``acquire`` into the at-capacity branch.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


class _ClosedEventStub:
    """``asyncio.Event`` stand-in whose cancelled ``wait()`` re-raises the
    chosen ``BaseException`` instead of ``CancelledError``, so the
    ``closed_task`` cleanup site in ``acquire`` surfaces it."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def clear(self) -> None:
        return None

    def set(self) -> None:
        return None

    def is_set(self) -> bool:
        return False

    async def wait(self) -> bool:
        # Park, then convert the cleanup's cancel into the chosen exception.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise self._exc from None
        return True  # pragma: no cover - never reached


def _force_at_capacity(pool: ConnectionPool, exc: BaseException) -> None:
    """Force ``acquire`` into the at-capacity queue-wait branch via
    ``_size = max_size`` and an empty queue, with the stub injecting the
    chosen exception on ``closed_event.wait()``."""
    pool._size = pool._max_size  # at capacity
    pool._closed_event = _ClosedEventStub(exc)  # type: ignore[assignment]


class _SentinelBaseException(BaseException):
    """``BaseException`` that bypasses pytest's special-cased
    KeyboardInterrupt/SystemExit handling for a clean propagation assert."""


@pytest.mark.asyncio
async def test_baseexception_during_cleanup_propagates() -> None:
    """A non-CancelledError ``BaseException`` from the cancelled-task drain
    must propagate; only ``CancelledError`` is meant to be drained."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1, timeout=5.0)
    _force_at_capacity(pool, _SentinelBaseException("signal"))

    with pytest.raises(_SentinelBaseException, match="signal"):
        async with pool.acquire():
            pass  # pragma: no cover - never reached
