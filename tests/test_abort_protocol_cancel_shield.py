"""Pin: ``DqliteConnection._abort_protocol`` shields its inner drain across outer
cancels so the StreamReader task is not orphaned ("Task was destroyed" at GC)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection

pytestmark = pytest.mark.asyncio


def _make_connection_with_stub_protocol_owning_inner_task(
    inner_running: asyncio.Event,
    inner_done: asyncio.Event,
) -> tuple[DqliteConnection, MagicMock, list[asyncio.Task[None]]]:
    """Build a connection whose ``wait_closed`` spawns a real inner ``asyncio.Task``.

    Without a real inner Task there is no orphan to observe, so the test could not
    distinguish pre-fix from post-fix behaviour.
    """
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._close_timeout = 5.0
    conn._address = "localhost:9001"
    proto = MagicMock()
    proto.close = MagicMock()

    inner_tasks: list[asyncio.Task[None]] = []

    async def _inner_body() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            inner_done.set()

    async def _slow_wait_closed() -> None:
        loop = asyncio.get_running_loop()
        inner = loop.create_task(_inner_body())
        inner_tasks.append(inner)
        inner_running.set()
        # Awaiting inner: an unobserved outer cancel here orphans the inner Task.
        await inner

    proto.wait_closed = AsyncMock(side_effect=_slow_wait_closed)
    conn._protocol = proto
    return conn, proto, inner_tasks


async def test_abort_protocol_outer_cancel_does_not_orphan_inner_task() -> None:
    """An ``asyncio.timeout`` fires while ``wait_closed`` awaits the inner Task;
    the shield + done-callback must keep it observed (no orphan warning at GC)."""
    from dqliteclient.cluster import _observe_drain_exception  # noqa: F401

    inner_running = asyncio.Event()
    inner_done = asyncio.Event()
    conn, proto, inner_tasks = _make_connection_with_stub_protocol_owning_inner_task(
        inner_running, inner_done
    )

    observer_invocations: list[asyncio.Task[None]] = []

    async def run_abort_under_timeout() -> None:
        # Patch on the cluster module: the production code's local import
        # re-reads the module attribute, so the spy is picked up.
        import dqliteclient.cluster as _cluster_mod

        original = _cluster_mod._observe_drain_exception

        def _spy(task: asyncio.Task[None]) -> None:
            observer_invocations.append(task)
            return original(task)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_cluster_mod, "_observe_drain_exception", _spy)
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await conn._abort_protocol()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await run_abort_under_timeout()
        # Cancel the inner sleep stub so it doesn't dangle for 10s.
        for t in inner_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for _ in range(10):
            await asyncio.sleep(0)
        gc.collect()

    pending_task_warnings = [
        str(w.message)
        for w in caught
        if "pending" in str(w.message).lower() or "destroyed" in str(w.message).lower()
    ]
    assert pending_task_warnings == [], (
        f"outer cancel must not orphan the inner drain task; got {pending_task_warnings!r}"
    )
    # Pin that the done-callback fired; fails without add_done_callback on the Task.
    assert observer_invocations, (
        "shield's done-callback discipline failed to invoke "
        "_observe_drain_exception on the inner drain Task"
    )
    proto.close.assert_called_once_with()


async def test_abort_protocol_with_no_protocol_returns_immediately() -> None:
    """A None ``_protocol`` makes the method a no-op."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._close_timeout = 5.0
    conn._address = "localhost:9001"
    conn._protocol = None
    await conn._abort_protocol()
    assert conn._protocol is None
