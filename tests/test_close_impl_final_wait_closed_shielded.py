"""Pin: ``DqliteConnection._close_impl``'s final
``wait_for(protocol.wait_closed())`` is wrapped in
``ensure_future + add_done_callback(_observe_drain_exception) +
shield`` — mirroring the discipline at ``_abort_protocol``.

Without the shield, an outer cancel landing mid-``wait_closed``
discards the underlying StreamReader task, surfacing later as a
"Task was destroyed but it is pending" diagnostic at GC. Direct
``await conn.close()`` callers (notably the dbapi-aio adapter's
``_async_conn.close()`` path) are NOT covered by ``__aexit__``'s
shield, so they are exposed without the discipline at this site.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_final_wait_closed_inner_task_not_cancelled_by_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shield's load-bearing property: a cancel landing while
    ``wait_closed`` is in flight must NOT cancel the inner drain task.
    The outer ``CancelledError`` propagates out of ``_close_impl``,
    but the inner ``wait_closed`` Task keeps running so the
    StreamReader's pending-task warning does not surface at GC.

    Pre-fix: ``wait_for(protocol.wait_closed(), ...)`` had no shield,
    so the inner task was cancelled along with the outer await.
    """
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

    # Capture the inner shielded task by hooking ``asyncio.ensure_future``
    # on the stdlib ``asyncio`` module — the module under test calls
    # ``asyncio.ensure_future(...)``, which resolves through the stdlib
    # binding at call time. (The Task type itself is C-implemented and
    # immutable, so we cannot patch ``add_done_callback`` directly.)
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

    # Load-bearing assertion: the inner shielded task is NOT
    # cancelled by the outer cancel.
    assert not inner.done(), (
        "inner wait_closed task must outlive outer cancel "
        "(shield keeps it running); the pre-fix unshielded "
        "wait_for would have cancelled it here"
    )

    # Release the parked drain so the inner task completes cleanly.
    finalise.set()
    await inner


@pytest.mark.asyncio
async def test_close_impl_final_wait_closed_observer_attached() -> None:
    """Pin the observer wiring: the inner ``wait_closed`` task must
    have ``_observe_drain_exception`` attached as a done-callback so
    eventual drain exceptions are surfaced (and not silently retained
    on the task as ``"Task exception was never retrieved"``)."""
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
