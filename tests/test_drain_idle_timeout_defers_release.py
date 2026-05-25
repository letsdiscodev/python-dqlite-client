"""Pin: ``_drain_idle``'s ``TimeoutError`` arm must defer the slot
release (``_release_reservation``) until the abandoned ``conn.close()``
task actually completes.

The pre-fix behaviour decremented ``_size`` in the finally arm
immediately on TimeoutError while the inner ``conn.close()`` task
continued running under the shield. A subsequent ``acquire()`` saw
``_size < max_size`` and dialed a new connection, transiently
exceeding the documented ``max_size`` cap until the orphan finished.

For pools sized against a hard cluster-side cap (e.g. Raft node
accepts N concurrent client connections), the transient overshoot
can drive new connects into a rejection envelope. Defer the slot
release to a follow-up coroutine that awaits the orphan drain before
calling ``_release_reservation`` — the ``max_size`` invariant the
docstring documents stays honoured end-to-end.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool

pytestmark = pytest.mark.asyncio


async def test_drain_idle_timeout_does_not_release_slot_until_drain_completes() -> None:
    """Drive ``_drain_idle`` with a queued connection whose ``close()``
    blocks past the per-iteration cap. Without the fix, ``_size`` is
    decremented immediately on TimeoutError. With the fix, ``_size``
    stays elevated until the orphan drain finishes.
    """
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._close_timeout = 0.01
    pool._closed = False
    pool._closed_event = None
    pool._lock = asyncio.Lock()
    pool._size = 1
    pool._max_size = 5
    pool._pool = asyncio.Queue()

    blocker = asyncio.Event()
    close_started = asyncio.Event()
    close_completed = asyncio.Event()

    class _StubConn:
        _address = "stub:9001"
        _pool_released = True

        async def close(self) -> None:
            close_started.set()
            try:
                await blocker.wait()
            finally:
                close_completed.set()

    stub = _StubConn()
    await pool._pool.put(stub)  # type: ignore[arg-type]

    drain_task = asyncio.create_task(pool._drain_idle())
    # Let close_started fire and the per-iteration wait_for fire
    # TimeoutError.
    await close_started.wait()
    # Wait for the drain loop to exit. With the fix, the finally
    # arm has deferred the slot release; with the old shape, it
    # has already decremented.
    await drain_task

    # Slot must still be reserved because the orphan close has not
    # completed yet.
    assert pool._size == 1, (
        "TimeoutError on per-iteration drain must defer the slot "
        f"release until the orphan close finishes; observed _size={pool._size}"
    )

    # Let the orphan drain finish. The deferred release fires.
    blocker.set()
    await close_completed.wait()
    # Yield a few times so the deferred release coroutine can run.
    for _ in range(50):
        if pool._size == 0:
            break
        await asyncio.sleep(0.01)

    assert pool._size == 0, (
        f"deferred slot release must fire once the orphan close finishes; "
        f"observed _size={pool._size}"
    )
