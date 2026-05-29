"""Without the shield, an outer cancel mid-``wait_closed`` discards the underlying
StreamReader task, surfacing as "Task was destroyed but it is pending" at GC; direct
``await conn.close()`` callers are not covered by ``__aexit__``'s shield."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_final_wait_closed_inner_task_not_cancelled_by_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel during ``wait_closed`` must not cancel the inner drain task: the outer
    CancelledError propagates but the inner Task keeps running."""
    loop = asyncio.get_running_loop()

    inner_entered = asyncio.Event()
    finalise = asyncio.Event()

    async def _parked_wait_closed() -> None:
        inner_entered.set()
        await finalise.wait()

    protocol = MagicMock()
    protocol.close = MagicMock()
    protocol.wait_closed = _parked_wait_closed

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = None
    conn._protocol = protocol
    conn._db_id = 7
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    import weakref

    conn._bound_loop_ref = weakref.ref(loop)
    conn._close_timeout = 5.0
    conn._address = "127.0.0.1:1"

    # Capture the inner shielded task via asyncio.ensure_future: the Task type is
    # C-implemented, so we cannot patch add_done_callback directly.
    captured_inner: list[asyncio.Task[object]] = []

    from typing import Any

    real_ensure_future = asyncio.ensure_future

    def _capturing_ensure_future(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        task = real_ensure_future(*args, **kwargs)
        captured_inner.append(task)
        return task

    monkeypatch.setattr(asyncio, "ensure_future", _capturing_ensure_future)
    close_task = loop.create_task(conn._close_impl())
    await inner_entered.wait()
    assert captured_inner, "the inner drain task must have been scheduled"
    inner = captured_inner[-1]

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert not inner.done(), (
        "inner wait_closed task must outlive outer cancel "
        "(shield keeps it running); the pre-fix unshielded "
        "wait_for would have cancelled it here"
    )

    finalise.set()
    await inner


@pytest.mark.asyncio
async def test_close_impl_final_wait_closed_observer_attached() -> None:
    """The inner ``wait_closed`` task must have ``_observe_drain_exception`` attached as a
    done-callback so drain exceptions are surfaced."""
    loop = asyncio.get_running_loop()

    observed_tasks: list[asyncio.Task[None]] = []

    from dqliteclient import cluster as cluster_mod

    real_observe = cluster_mod._observe_drain_exception

    def _capturing_observer(task: asyncio.Task[None]) -> None:
        observed_tasks.append(task)
        real_observe(task)

    async def _quick_drain() -> None:
        return None

    protocol = MagicMock()
    protocol.close = MagicMock()
    protocol.wait_closed = _quick_drain

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = None
    conn._protocol = protocol
    conn._db_id = 7
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    import weakref

    conn._bound_loop_ref = weakref.ref(loop)
    conn._close_timeout = 5.0
    conn._address = "127.0.0.1:1"

    cluster_mod._observe_drain_exception = _capturing_observer  # type: ignore[assignment]
    try:
        await conn._close_impl()
    finally:
        cluster_mod._observe_drain_exception = real_observe

    assert observed_tasks, (
        "the final wait_closed drain task must have "
        "_observe_drain_exception attached as a done-callback"
    )
