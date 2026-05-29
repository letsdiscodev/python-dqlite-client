"""Pin: the drain paths schedule the inner ``conn.close()`` as a Task with an
``_observe_drain_exception`` done-callback before the shielded ``wait_for``, so
an abandoned close that later raises does not log "Task exception was never
retrieved" at GC."""

from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

pytestmark = pytest.mark.asyncio


async def test_pool_drain_idle_abandoned_close_does_not_orphan_task() -> None:
    """A close that hangs past the cap then raises: the done-callback must
    observe it so no GC warning emerges."""
    from dqliteclient.pool import ConnectionPool

    pool = ConnectionPool.__new__(ConnectionPool)
    pool._close_timeout = 0.01
    pool._closed = False

    blocker = asyncio.Event()

    class _StubConn:
        _address = "stub:9001"
        _pool_released = False

        async def close(self) -> None:
            await blocker.wait()
            raise RuntimeError("simulated post-abandon transport raise")

    stub = _StubConn()

    # Exercise the drain's shape directly: ensure_future +
    # add_done_callback(_observe_drain_exception) + wait_for(shield(...)).
    from dqliteclient.cluster import _observe_drain_exception

    inner_drain = asyncio.ensure_future(stub.close())
    inner_drain.add_done_callback(_observe_drain_exception)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(inner_drain), timeout=0.01)
        # Release the blocker so close() raises and the callback observes it.
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
