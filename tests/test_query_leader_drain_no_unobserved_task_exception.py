"""Pin: the leader-probe finally writer drain does not leak an unobserved
TimeoutError from the inner shielded Task on outer cancel. A done-callback
observes ``.exception()`` so asyncio's GC-time warning never fires."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from dqliteclient import cluster as cluster_module


def test_observe_drain_exception_reaps_timeout_error() -> None:
    """Functional pin: the helper consumes the exception so a later
    ``t.exception()`` read does not re-raise."""

    async def _drive() -> None:
        async def _slow() -> None:
            await asyncio.sleep(60)

        inner = asyncio.ensure_future(asyncio.wait_for(_slow(), timeout=0.001))
        with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
            await inner
        cluster_module._observe_drain_exception(inner)
        assert isinstance(inner.exception(), TimeoutError)

    asyncio.run(_drive())


def test_observe_drain_exception_skips_cancelled_task() -> None:
    """Functional pin: the helper is no-op-safe on a cancelled task; it must not raise."""

    async def _drive() -> None:
        async def _slow() -> None:
            await asyncio.sleep(60)

        cancelled = asyncio.create_task(_slow())
        cancelled.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancelled
        assert cancelled.cancelled(), "test setup: task must be cancelled, not resolved"
        cluster_module._observe_drain_exception(cancelled)

    asyncio.run(_drive())


@pytest.mark.asyncio
async def test_outer_cancel_during_drain_does_not_emit_unobserved_warning() -> None:
    """Behavioural pin: cancel outer mid-drain, let inner reach TimeoutError, and
    verify no "Task exception was never retrieved" diagnostic fires. asyncio surfaces
    that diagnostic via ``loop.call_exception_handler``, not ``warnings.warn``."""

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
    """Positive control: with no done-callback, the diagnostic IS emitted, proving
    the capture mechanism works (so the negative pin can't pass for the wrong reason)."""

    async def _slow() -> None:
        await asyncio.sleep(60)

    captured: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        # No done-callback, so the resolved-with-TimeoutError task stays unobserved.
        inner = asyncio.ensure_future(asyncio.wait_for(_slow(), timeout=0.001))
        await asyncio.sleep(0.05)
        # Drop the reference so GC finalises the task and reports the unobserved exception.
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
