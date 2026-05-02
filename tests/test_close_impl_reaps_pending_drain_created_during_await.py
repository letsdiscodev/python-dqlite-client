"""Pin: ``DqliteConnection._close_impl`` reaps a fresh
``_pending_drain`` task that a concurrent ``_invalidate`` callback
created during the prior drain's ``await pending``.

Without the bounded re-snapshot loop, a
``loop.call_soon_threadsafe(self._invalidate, ...)`` callback queued
by the dbapi sync wrapper's timeout / KI arms can run during
``await pending``, find ``self._protocol`` still non-None, and create
a NEW ``_pending_drain`` task. After the original ``await pending``
resolves, ``_close_impl`` reads ``self._protocol`` (now None — the
racing ``_invalidate`` cleared it), short-circuits, and returns —
leaving the new drain task orphaned. asyncio emits
"Task was destroyed but it is pending" at interpreter shutdown.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_drains_late_arriving_pending_drain() -> None:
    """Drive the race shape: snapshot pending=A; during ``await A``,
    a callback creates pending=B (a still-running task). The bounded
    re-snapshot loop must observe B and await it, leaving no orphan.
    """
    loop = asyncio.get_running_loop()

    async def _slow_drain(label: str) -> None:
        # Survive long enough that close_impl observes us as not done.
        await asyncio.sleep(0.005)

    pending_a = loop.create_task(_slow_drain("A"))
    pending_b_holder: list[asyncio.Task[None]] = []

    def _race_invalidate_callback() -> None:
        # Simulates the dbapi sync wrapper's
        # ``loop.call_soon_threadsafe(dying._invalidate, ...)`` that
        # runs while close_impl is awaiting the prior drain.
        b = loop.create_task(_slow_drain("B"))
        pending_b_holder.append(b)
        conn._pending_drain = b

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = pending_a
    conn._protocol = None  # short-circuit the rest of close_impl
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop = None

    # Schedule the race callback to fire while we're inside the
    # await pending block. ``call_soon`` runs at the next loop
    # iteration which is exactly when ``await pending`` yields.
    loop.call_soon(_race_invalidate_callback)

    # asyncio surfaces the "Task was destroyed but it is pending" /
    # "Task exception was never retrieved" diagnostics via the loop's
    # exception handler, NOT via ``warnings.warn``. Capture via
    # ``loop.set_exception_handler`` so the assert sees what asyncio
    # actually emits.
    captured: list[dict[str, object]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        await conn._close_impl()
        # Let the loop drain so any orphaned task surfaces a diagnostic
        # at GC.
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(prior_handler)

    # Both A and B must be done — the loop reaped both.
    assert pending_a.done()
    assert pending_b_holder, "race callback should have run"
    assert pending_b_holder[0].done(), (
        "fresh _pending_drain task created during await must be reaped"
    )
    # No "Task was destroyed but it is pending" diagnostic was
    # emitted via asyncio's exception handler.
    asyncio_diagnostics = [
        ctx for ctx in captured if "Task was destroyed" in str(ctx.get("message", ""))
    ]
    if asyncio_diagnostics:
        msgs = [ctx.get("message") for ctx in asyncio_diagnostics]
        raise AssertionError(f"Expected no orphaned-task diagnostics; got {msgs}")


@pytest.mark.asyncio
async def test_close_impl_fails_loud_when_resnapshot_cap_exhausted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pathological feedback loop: a racing callback re-creates a
    fresh ``_pending_drain`` on every iteration. The cap exhausts;
    the loop must (a) cancel the residual task to avoid the
    "Task was destroyed but it is pending" diagnostic at GC, and
    (b) log a WARNING so operators see the loop in production logs.

    Pre-fix the loop's ``else:`` clause did not exist — the loop
    silently broke with ``_pending_drain`` still pointing at a live
    task and the warning the comment promised never fired.
    """
    import logging

    loop = asyncio.get_running_loop()

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._protocol = None
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop = None

    # Adversarial: each ``await pending`` that resolves triggers a
    # fresh not-done task to be assigned to ``conn._pending_drain``.
    # Use a custom awaitable so we can precisely simulate the
    # pathological _invalidate feedback loop: the awaitable's
    # ``__await__`` yields once (giving control back to _close_impl),
    # then sets a fresh adversary on conn._pending_drain before
    # returning. This guarantees iteration N+1 ALWAYS sees a
    # not-done pending, exhausting the cap.
    fresh_tasks: list[asyncio.Task[None]] = []

    from collections.abc import Generator
    from typing import Any

    class _Adversary:
        def __init__(self) -> None:
            self._real = loop.create_task(asyncio.sleep(0.001))
            fresh_tasks.append(self._real)

        def done(self) -> bool:
            return False  # always reports not-done so loop awaits

        def cancel(self) -> bool:
            return self._real.cancel()

        def __await__(self) -> Generator[Any]:
            yield from self._real.__await__()
            # Before returning, plant a fresh adversary so iteration
            # N+1 sees a non-done pending.
            conn._pending_drain = _Adversary()  # type: ignore[assignment]

    conn._pending_drain = _Adversary()  # type: ignore[assignment]

    captured: list[dict[str, object]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    with caplog.at_level(logging.WARNING, logger="dqliteclient.connection"):
        try:
            await conn._close_impl()
            await asyncio.sleep(0.05)
        finally:
            loop.set_exception_handler(prior_handler)

    # The fail-loud WARNING must fire when the cap is exhausted.
    warnings_seen = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "re-snapshot iterations" in r.getMessage()
    ]
    assert warnings_seen, (
        "_close_impl must log a WARNING when the bounded re-snapshot loop's "
        "cap is exhausted; without it, the comment's 'fail loudly' promise "
        "is unkept and operators have no signal of the feedback loop."
    )
    # The residual pending_drain must be cleared so no orphaned-task
    # diagnostic surfaces.
    assert conn._pending_drain is None
    asyncio_diagnostics = [
        ctx for ctx in captured if "Task was destroyed" in str(ctx.get("message", ""))
    ]
    if asyncio_diagnostics:
        msgs = [ctx.get("message") for ctx in asyncio_diagnostics]
        raise AssertionError(f"Expected no orphaned-task diagnostics; got {msgs}")


@pytest.mark.asyncio
async def test_close_impl_loop_terminates_when_invalidate_clears_protocol() -> None:
    """The re-snapshot loop terminates after one iteration when the
    racing ``_invalidate`` clears ``_protocol`` (the production
    callback path) — no infinite spin."""
    loop = asyncio.get_running_loop()

    async def _slow(label: str) -> None:
        await asyncio.sleep(0.001)

    pending_a = loop.create_task(_slow("A"))

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = pending_a
    conn._protocol = None
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop = None

    await conn._close_impl()
    assert pending_a.done()
    assert conn._pending_drain is None
