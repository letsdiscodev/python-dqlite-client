"""Pin: ``DqliteConnection._abort_protocol`` shields its inner drain
across outer cancels so the underlying StreamReader task is not
orphaned.

An outer cancel landing while ``wait_closed`` is in flight
previously cancelled the drain without awaiting the inner task to
completion — surfacing as ``"Task was destroyed but it is pending"``
warnings at GC. The shield+done-callback discipline (mirroring the
sibling ``_connect_impl`` finally arm and
``ClusterClient.open_admin_connection``'s drain) keeps the inner
task alive and observed.
"""

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
    """Build a connection whose ``protocol.wait_closed`` body spawns a
    real ``asyncio.Task`` (the orphan-class of bug the shield+
    done-callback discipline targets).

    Without a real inner Task, ``asyncio.wait_for(coro)`` cancellation
    propagates cleanly to the awaiter with no orphan and the test
    cannot distinguish the pre-fix shape from the post-fix shape.
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
        # Await the inner Task: the orphan-class manifests if the
        # outer cancel discards this await without observing the
        # inner Task. With shield + done-callback the inner is
        # owned and observed; without, the inner would be left as
        # "Task was destroyed but it is pending" at GC.
        await inner

    proto.wait_closed = AsyncMock(side_effect=_slow_wait_closed)
    conn._protocol = proto
    return conn, proto, inner_tasks


async def test_abort_protocol_outer_cancel_does_not_orphan_inner_task() -> None:
    """Schedule ``_abort_protocol`` inside an ``asyncio.timeout`` that
    fires while the stub's ``wait_closed`` is awaiting a real inner
    ``asyncio.Task``. The shield + done-callback discipline must
    keep the inner Task observed so no ``"Task was destroyed but
    it is pending"`` warning lands at GC.
    """
    from dqliteclient.cluster import _observe_drain_exception  # noqa: F401

    inner_running = asyncio.Event()
    inner_done = asyncio.Event()
    conn, proto, inner_tasks = _make_connection_with_stub_protocol_owning_inner_task(
        inner_running, inner_done
    )

    observer_invocations: list[asyncio.Task[None]] = []

    async def run_abort_under_timeout() -> None:
        # Spy on ``_observe_drain_exception`` so the test can assert
        # the done-callback actually fired on the inner Task. Patch
        # the imported symbol on the cluster module — the production
        # code's local-import resolves to the patched object because
        # ``from dqliteclient.cluster import _observe_drain_exception``
        # binds to the current module-attribute value at import time.
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
        # Cancel the spawned inner Task explicitly (the production
        # close paths normally tear it down via the surrounding
        # connection finalisers; in this test the inner is a sleep
        # stub that would otherwise dangle for 10s).
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
    # Done-callback discipline pin: the spy was invoked on the
    # shield-wrapped Task. Without the explicit
    # ``add_done_callback(_observe_drain_exception)`` at
    # connection.py:2068, this assertion fires.
    assert observer_invocations, (
        "shield's done-callback discipline failed to invoke "
        "_observe_drain_exception on the inner drain Task"
    )
    # Sync ``proto.close()`` tear-down still ran.
    proto.close.assert_called_once_with()


async def test_abort_protocol_with_no_protocol_returns_immediately() -> None:
    """If ``_protocol`` is None (already aborted / never connected),
    the method is a no-op. Preserved across the shield migration.
    """
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._close_timeout = 5.0
    conn._address = "localhost:9001"
    conn._protocol = None
    await conn._abort_protocol()
    # ``conn._protocol`` was None to begin with; the early return
    # leaves it untouched.
    assert conn._protocol is None
