"""Pin: the leader-probe ``finally`` writer drain does NOT leak an
unobserved ``TimeoutError`` from the inner ``wait_for`` Task on outer
cancel.

The drain wraps ``writer.wait_closed()`` in
``asyncio.shield(asyncio.wait_for(...))``. When the awaiter is
cancelled mid-shield, ``shield`` re-raises CancelledError to the
awaiter while letting the inner Task survive. If the inner Task
later resolves with TimeoutError (peer unresponsive) and no coroutine
is awaiting the Task, asyncio's task-finalisation logger emits
``"Task exception was never retrieved"`` at GC. The fix is an
explicit done-callback that observes ``.exception()`` so the
warning never fires.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect

import pytest

from dqliteclient import cluster as cluster_module


def test_observe_drain_exception_is_module_level_helper() -> None:
    """Source-level pin: the helper exists. A regression that inlines
    it back without observation would surface here."""
    assert hasattr(cluster_module, "_observe_drain_exception")
    assert callable(cluster_module._observe_drain_exception)


def test_observe_drain_exception_reaps_timeout_error() -> None:
    """Functional pin: the helper consumes the exception so a future
    ``t.exception()`` read does not re-raise."""

    async def _drive() -> None:
        async def _slow() -> None:
            await asyncio.sleep(60)

        # Build a task that resolves with TimeoutError.
        inner = asyncio.ensure_future(asyncio.wait_for(_slow(), timeout=0.001))
        # Wait for the inner to time out.
        with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
            await inner
        # Inner is done with TimeoutError — observe via the helper.
        cluster_module._observe_drain_exception(inner)
        # ``.exception()`` on an already-observed task does not re-raise.
        assert isinstance(inner.exception(), TimeoutError)

    asyncio.run(_drive())


def test_query_leader_finally_uses_observed_drain_pattern() -> None:
    """Source-level pin: the leader-probe finally wraps the inner
    drain in ``asyncio.ensure_future`` plus the
    ``add_done_callback(_observe_drain_exception)`` pattern, NOT the
    direct ``shield(wait_for(...))`` composition that orphans the
    inner Task."""
    src = inspect.getsource(cluster_module)
    # The shield(...) call must consume an explicit Task variable, not
    # an inline asyncio.wait_for(...) — that's the regression shape.
    assert "asyncio.shield(inner_drain)" in src, (
        "Leader-probe finally must shield an explicit inner-Task variable "
        "with an exception-observer done-callback (NOT a composed "
        "shield(wait_for(...)))"
    )
    assert "add_done_callback(_observe_drain_exception)" in src


@pytest.mark.asyncio
async def test_outer_cancel_during_drain_does_not_emit_unobserved_warning() -> None:
    """Behavioural pin: cancel the outer task mid-drain, let the inner
    task run to its TimeoutError, drain the loop, and verify NO
    "Task exception was never retrieved" diagnostic fires.

    Drives a synthetic case mirroring the production shape — a task
    that runs the same shielded pattern in a tight loop, with the
    outer cancelled mid-flight.

    asyncio surfaces the unobserved-exception diagnostic via
    ``loop.call_exception_handler``, NOT via ``warnings.warn``.
    Capture via the loop's exception handler so the assert sees what
    asyncio actually emits.
    """

    async def _slow() -> None:
        await asyncio.sleep(60)

    async def _drain_under_shield() -> None:
        inner = asyncio.ensure_future(asyncio.wait_for(_slow(), timeout=0.05))
        inner.add_done_callback(cluster_module._observe_drain_exception)
        with contextlib.suppress(OSError, TimeoutError):
            await asyncio.shield(inner)

    captured: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        t = asyncio.create_task(_drain_under_shield())
        await asyncio.sleep(0.001)
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
        # Let the inner timeout fire and finalise.
        await asyncio.sleep(0.1)
    finally:
        loop.set_exception_handler(prior_handler)

    # Ensure no asyncio "Task exception was never retrieved" diagnostic
    # was emitted to the loop's exception handler.
    asyncio_diagnostics = [
        ctx
        for ctx in captured
        if "Task exception was never retrieved" in str(ctx.get("message", ""))
    ]
    if asyncio_diagnostics:
        msgs = [ctx.get("message") for ctx in asyncio_diagnostics]
        raise AssertionError(f"Expected no unobserved-task diagnostics; got {msgs}")


@pytest.mark.asyncio
async def test_loop_exception_handler_captures_unobserved_task_exception() -> None:
    """Positive control: with no done-callback, the unobserved-
    exception diagnostic IS emitted via ``loop.call_exception_handler``
    — verifying the capture mechanism works. Without this control, a
    future regression on the capture path itself (e.g., asyncio
    changing how it surfaces the diagnostic) would silently make the
    negative pin above pass for the wrong reason.
    """

    async def _slow() -> None:
        await asyncio.sleep(60)

    captured: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        # Build an inner task that resolves with TimeoutError —
        # crucially with NO done-callback to observe the exception.
        inner = asyncio.ensure_future(asyncio.wait_for(_slow(), timeout=0.001))
        # Wait for the timeout to fire without awaiting the inner.
        await asyncio.sleep(0.05)
        # Drop the reference so the task can be GC'd → ``__del__``
        # fires → ``call_exception_handler`` reports the unobserved
        # exception.
        del inner
        import gc

        gc.collect()
        await asyncio.sleep(0.01)
    finally:
        loop.set_exception_handler(prior_handler)

    asyncio_diagnostics = [
        ctx
        for ctx in captured
        if "Task exception was never retrieved" in str(ctx.get("message", ""))
    ]
    assert asyncio_diagnostics, (
        "Capture mechanism must observe asyncio's unobserved-exception "
        "diagnostic; if this fails, the negative pin above is silently "
        "passing for the wrong reason."
    )
