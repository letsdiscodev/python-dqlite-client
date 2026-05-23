"""Pin reservation accounting across three subtle paths in
``ConnectionPool.acquire`` and ``ConnectionPool._release``:

* The fresh-slot reservation increment lives INSIDE the same try-
  frame whose except arm releases it. The pre-fix shape opened the
  ``try:`` only AFTER ``async with self._lock:`` exited — leaving a
  bytecode window between lock-release and try-frame in which a
  ``BaseException`` (e.g., ``KeyboardInterrupt`` from a SIGINT
  delivered to the foreground task) could escape with ``_size``
  incremented but no compensating decrement. Mirrors the
  ``DqliteConnection._run_protocol`` discipline (assignment INSIDE
  the existing try-frame, not after lock-exit).

* Both ``_create_connection`` call sites (the fresh-slot path AND
  the dead-conn-replacement arm) honour the user-supplied
  ``pool.timeout`` deadline. The pre-fix shape ran
  ``_create_connection`` unbounded, so a saturated pool with
  ``timeout=0.1`` could wait up to the cluster.connect retry
  budget (~30 s default) before failing. The user-visible
  ``acquire()`` contract is "return within ``timeout``"; the
  implementation must clamp the create-connection await to the
  remaining budget.

* ``_release(conn)`` short-circuits in a forked child process
  rather than touching the parent's loop-bound ``self._lock``
  (which raises a confusing asyncio-internal ``RuntimeError``).
  Mirrors the fork short-circuit already in ``acquire()`` and
  ``close()``.
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
    """The bytecode-window pin: if anything between ``self._size += 1``
    and the protective ``try:`` raises, the reservation must NOT leak.

    Inject the failure at the lock's ``__aexit__`` so the increment
    has already landed but the user-level ``try:`` envelope is the
    only thing that can clean up. With the pre-fix shape (``try:``
    opened AFTER ``async with self._lock:``), the increment leaked.
    """
    pool = ConnectionPool(["localhost:9001"], max_size=2, timeout=1.0)

    # Replace ``_lock`` with a subclass whose ``__aexit__`` raises
    # AFTER the parent class releases the lock. ``async with`` dispatch
    # goes through ``type(obj).__aexit__`` per PEP 492, so the lock's
    # type — not the instance — must carry the patched method.
    class _RaisingLock(asyncio.Lock):
        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            await super().__aexit__(exc_type, exc, tb)
            raise RuntimeError("synthetic post-lock-exit failure")

    raising_lock = _RaisingLock()
    pool._lock = raising_lock

    with pytest.raises(RuntimeError, match="synthetic post-lock-exit failure"):
        async with pool.acquire():
            pytest.fail("should not reach the with-body")

    # The decrement MUST have run — the reservation is back at zero.
    assert pool._size == 0, (
        f"reservation leaked across the bytecode window: _size={pool._size}, "
        f"expected 0 after the synthetic failure"
    )


@pytest.mark.asyncio
async def test_acquire_create_connection_clamped_by_pool_timeout() -> None:
    """A saturated pool with ``timeout=0.1`` must NOT block longer
    than ``timeout`` waiting for ``_create_connection``. The
    cluster.connect retry budget (~30 s default) must be clamped
    to the remaining acquire deadline.
    """
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)

    async def _slow_create() -> Any:
        # Sleeps far longer than pool.timeout so a clamp is observable.
        await asyncio.sleep(2.0)
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
    # Leave a generous tolerance band for slow CI runners; the cap
    # we want to refute is "could wait minutes." Anything well under
    # the 2 s sleep proves the clamp is present.
    assert elapsed < 0.6, (
        f"acquire() blocked for {elapsed:.3f}s under pool.timeout=0.1s; "
        f"_create_connection ran past the pool deadline"
    )


@pytest.mark.asyncio
async def test_release_short_circuits_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_release(conn)`` must short-circuit in the forked child
    rather than acquiring the parent-loop-bound ``self._lock``.

    Without the guard, a ``RuntimeError("got Future ... attached
    to a different loop")`` escapes from the child's
    ``_release_reservation`` call, masking the canonical
    ``InterfaceError("Pool used after fork...")`` diagnostic that
    ``acquire`` and ``close`` produce on the same misuse.
    """
    pool = ConnectionPool(["localhost:9001"], max_size=2, timeout=1.0)

    # Build a minimal fake connection: the release path checks
    # ``conn.is_connected`` and ``_pool_released`` — the fork
    # short-circuit must trip before any of that runs.
    class _FakeConn:
        is_connected = True
        _pool_released = False

    fake = _FakeConn()

    # Simulate "we are the forked child" by flipping the cached pid.
    _real_getpid = os.getpid
    monkeypatch.setattr("dqliteclient.connection.os.getpid", lambda: _real_getpid() + 1)

    # Must NOT raise RuntimeError from asyncio internals.
    await pool._release(fake)  # type: ignore[arg-type]

    # The conn must be marked released so its finalizer doesn't
    # surface a spurious ResourceWarning at GC.
    assert fake._pool_released is True, (
        "fork short-circuit must mark the conn as released so the "
        "ResourceWarning finalizer stays quiet at GC time"
    )
    # The pool's accounting must be untouched (no decrement; the
    # parent still owns the slot).
    assert pool._size == 0
