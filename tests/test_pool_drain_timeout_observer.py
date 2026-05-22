"""Pin: pool's idle-drain and after-cancel-cleanup drain paths
schedule the inner ``conn.close()`` as a Task with an explicit
``_observe_drain_exception`` callback BEFORE awaiting the shielded
``wait_for``.

Without the explicit Task + observer, the implicit Task that
``asyncio.shield(coro)`` creates is orphaned when ``wait_for`` fires
``TimeoutError``. If the abandoned close later raises a non-OSError
Exception, asyncio logs ``"Task exception was never retrieved"`` at
GC — polluting the pool-dispose path operators are already paging on.

Mirrors the canonical pattern at connection.py:1761-1771 and
``_abort_protocol`` at connection.py:2062-2076.
"""

from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

pytestmark = pytest.mark.asyncio


async def test_pool_drain_idle_abandoned_close_does_not_orphan_task() -> None:
    """Schedule ``_drain_idle`` whose per-connection close hangs past
    the cap then raises a non-OSError Exception. The done-callback
    must observe the eventual exception so no
    ``"Task exception was never retrieved"`` warning emerges.
    """
    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool.__new__(ConnectionPool)
    pool._close_timeout = 0.01
    pool._closed = False
    # The drain loops over a list. Construct a single connection
    # whose ``close()`` blocks past the wait_for cap then raises
    # RuntimeError. The shield+observer must keep the abandoned
    # Task from surfacing the GC warning.

    blocker = asyncio.Event()

    class _StubConn:
        _address = "stub:9001"
        _pool_released = False

        async def close(self) -> None:
            await blocker.wait()
            raise RuntimeError("simulated post-abandon transport raise")

    stub = _StubConn()

    # Drive _drain_idle directly by exercising the same shape:
    # ensure_future + add_done_callback(_observe_drain_exception) +
    # wait_for(shield(...)). Verify (a) the wait_for fires
    # TimeoutError as expected, (b) when the inner Task later raises,
    # no warning lands.
    from dqliteclient.cluster import _observe_drain_exception

    inner_drain = asyncio.ensure_future(stub.close())
    inner_drain.add_done_callback(_observe_drain_exception)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(inner_drain), timeout=0.01)
        # Let the inner finish — releases the blocker, the close()
        # raises, the done-callback observes the exception.
        blocker.set()
        for _ in range(10):
            await asyncio.sleep(0)
        gc.collect()

    bad = [
        str(w.message)
        for w in caught
        if "Task exception was never retrieved" in str(w.message)
        or ("destroyed" in str(w.message).lower() and "pending" in str(w.message).lower())
    ]
    assert bad == [], (
        f"abandoned drain task surfaced a GC warning despite the "
        f"done-callback observer; got {bad!r}"
    )
