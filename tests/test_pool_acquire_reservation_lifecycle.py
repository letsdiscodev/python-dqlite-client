"""Pin reservation accounting across three subtle paths in
``ConnectionPool.acquire`` and ``ConnectionPool._release``:

* The fresh-slot ``_size`` increment must live inside the try-frame whose
  except arm releases it; a window between lock-release and try-open could
  leak the reservation if a BaseException (e.g. SIGINT) escaped.
* Both ``_create_connection`` call sites must clamp to the remaining
  ``pool.timeout`` rather than running the unbounded ~30s connect budget.
* ``_release`` must short-circuit in a forked child rather than touch the
  parent-loop-bound ``_lock`` (which raises a confusing asyncio RuntimeError).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_reserved_slot_released_when_lock_exit_raises() -> None:
    """If anything between ``_size += 1`` and the protective ``try:`` raises,
    the reservation must not leak. The failure is injected at the lock's
    ``__aexit__``, after the increment lands."""
    pool = ConnectionPool(["localhost:9001"], max_size=2, timeout=1.0)

    # __aexit__ raises after releasing the lock. async with dispatches
    # through type(obj).__aexit__, so patch the type, not the instance.
    class _RaisingLock(asyncio.Lock):
        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            await super().__aexit__(exc_type, exc, tb)
            raise RuntimeError("synthetic post-lock-exit failure")

    raising_lock = _RaisingLock()
    pool._lock = raising_lock

    with pytest.raises(RuntimeError, match="synthetic post-lock-exit failure"):
        async with pool.acquire():
            pytest.fail("should not reach the with-body")

    assert pool._size == 0, (
        f"reservation leaked across the bytecode window: _size={pool._size}, "
        f"expected 0 after the synthetic failure"
    )


@pytest.mark.asyncio
async def test_acquire_create_connection_clamped_by_pool_timeout() -> None:
    """A saturated pool with ``timeout=0.1`` must clamp ``_create_connection``
    (the ~30s connect budget) to the remaining acquire deadline."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)  # far past pool.timeout so the clamp shows
        raise AssertionError("should have been clamped by pool.timeout")

    loop = asyncio.get_running_loop()
    started = loop.time()

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        pytest.raises(DqliteConnectionError),
    ):
        async with pool.acquire():
            pytest.fail("should not reach the with-body")

    elapsed = loop.time() - started
    # Generous band for slow CI; well under the 2s sleep proves the clamp.
    assert elapsed < 0.6, (
        f"acquire() blocked for {elapsed:.3f}s under pool.timeout=0.1s; "
        f"_create_connection ran past the pool deadline"
    )


@pytest.mark.asyncio
async def test_release_short_circuits_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_release`` must short-circuit in the forked child rather than touch
    the parent-loop-bound ``_lock``, whose RuntimeError would mask the
    canonical ``InterfaceError("Pool used after fork...")`` diagnostic."""
    pool = ConnectionPool(["localhost:9001"], max_size=2, timeout=1.0)

    # The fork short-circuit must trip before the release path reads
    # conn.is_connected / _pool_released.
    class _FakeConn:
        is_connected = True
        _pool_released = False

    fake = _FakeConn()

    # Simulate the forked child by flipping the cached pid.
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)

    # Must NOT raise RuntimeError from asyncio internals.
    await pool._release(fake)  # type: ignore[arg-type]

    assert fake._pool_released is True, (
        "fork short-circuit must mark the conn as released so the "
        "ResourceWarning finalizer stays quiet at GC time"
    )
    # Accounting untouched: the parent still owns the slot.
    assert pool._size == 0
