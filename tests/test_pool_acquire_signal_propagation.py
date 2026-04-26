"""``ConnectionPool.acquire`` cleanup must not swallow KeyboardInterrupt / SystemExit.

The four cancel-and-drain-task sites in ``acquire()`` (around the
``closed_task`` / ``get_task`` cleanup) used to wrap the await in
``contextlib.suppress(BaseException)``. The intent was to drain the
cancelled task's ``CancelledError`` without re-raising. ``BaseException``
is too broad: it also catches ``KeyboardInterrupt`` and ``SystemExit``,
which the Python signal-propagation contract requires to bubble out.

Pin the narrowed semantics: ``CancelledError`` from the cancelled await
is silenced (existing happy-path); ``KeyboardInterrupt`` / ``SystemExit``
from the await propagates.

The tests intercept ``closed_event.wait()`` coroutines so the
``closed_task`` produced by ``acquire`` raises the chosen exception
when awaited during cleanup. The pool is pre-loaded so ``acquire``
takes the at-capacity branch where the four sites live.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


class _ClosedEventStub:
    """Stand-in for ``asyncio.Event`` whose cancelled ``wait()`` re-raises
    a ``BaseException`` instead of ``CancelledError``.

    The pool stores its closed-event under ``_closed_event``;
    swapping it out makes ``closed_task = asyncio.create_task(
    closed_event.wait())`` produce a task whose cancellation surfaces
    as the chosen ``BaseException``. The cleanup site in ``acquire``
    cancels then awaits this task — the exact path under test.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def clear(self) -> None:
        return None

    def set(self) -> None:
        return None

    def is_set(self) -> bool:
        return False

    async def wait(self) -> bool:
        # Park indefinitely. When the cleanup cancels this coroutine,
        # convert the CancelledError into the test's chosen BaseException
        # so we can assert it propagates out of ``acquire``.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise self._exc from None
        return True  # pragma: no cover - never reached


def _force_at_capacity(pool: ConnectionPool, exc: BaseException) -> None:
    """Make ``acquire`` enter the at-capacity ``asyncio.wait`` branch.

    The pool reserves the first slot via ``_size`` accounting; pre-
    setting ``_size = max_size`` plus an empty in-queue forces the
    queue-wait path. The closed-event stub injects the chosen
    exception when ``acquire`` awaits ``closed_event.wait()``.
    """
    pool._size = pool._max_size  # at capacity
    pool._closed_event = _ClosedEventStub(exc)  # type: ignore[assignment]


class _SentinelBaseException(BaseException):
    """Custom ``BaseException`` (not ``Exception``) that bypasses
    pytest's special-cased KeyboardInterrupt/SystemExit handling so
    we can assert ``BaseException``-class propagation cleanly."""


@pytest.mark.asyncio
async def test_baseexception_during_cleanup_propagates() -> None:
    """``KeyboardInterrupt`` / ``SystemExit`` and any other
    ``BaseException`` from the cancelled-task drain must propagate.
    The cleanup may not silently swallow them — only ``CancelledError``
    is meant to be drained."""
    pool = ConnectionPool("localhost:9001", min_size=0, max_size=1, timeout=5.0)  # type: ignore[arg-type]
    _force_at_capacity(pool, _SentinelBaseException("signal"))

    with pytest.raises(_SentinelBaseException, match="signal"):
        async with pool.acquire():
            pass  # pragma: no cover - never reached
