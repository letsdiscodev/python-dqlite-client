"""The bounded re-snapshot loop reaps a fresh ``_pending_drain`` that a concurrent
``_invalidate`` callback created during the prior drain's ``await pending``; without it the
new drain task is orphaned ("Task was destroyed but it is pending" at shutdown)."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_drains_late_arriving_pending_drain() -> None:
    """Snapshot pending=A; during ``await A`` a callback creates pending=B; the loop must
    observe B and await it, leaving no orphan."""
    loop = asyncio.get_running_loop()

    async def _slow_drain(label: str) -> None:
        await asyncio.sleep(0.005)

    pending_a = loop.create_task(_slow_drain("A"))
    pending_b_holder: list[asyncio.Task[None]] = []

    def _race_invalidate_callback() -> None:
        # Simulates the dbapi sync wrapper's call_soon_threadsafe(_invalidate, ...)
        # running while close_impl awaits the prior drain.
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
    conn._bound_loop_ref = None

    # call_soon fires the race callback at the next loop iteration, exactly when
    # ``await pending`` yields.
    loop.call_soon(_race_invalidate_callback)

    # asyncio surfaces orphaned-task diagnostics via the loop exception handler, not
    # warnings.warn, so capture via set_exception_handler.
    captured: list[dict[str, object]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        await conn._close_impl()
        # Drain the loop so any orphaned task surfaces a diagnostic.
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(prior_handler)

    assert pending_a.done()
    assert pending_b_holder, "race callback should have run"
    assert pending_b_holder[0].done(), (
        "fresh _pending_drain task created during await must be reaped"
    )
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
    """When a racing callback re-creates ``_pending_drain`` every iteration, the cap
    exhausts; the loop must cancel the residual task and log a WARNING."""
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
    conn._bound_loop_ref = None

    # Custom awaitable whose __await__ yields once then plants a fresh adversary on
    # conn._pending_drain, guaranteeing iteration N+1 always sees a not-done pending.
    fresh_tasks: list[asyncio.Task[None]] = []

    from collections.abc import Generator
    from typing import Any

    class _Adversary:
        def __init__(self) -> None:
            self._real = loop.create_task(asyncio.sleep(0.001))
            fresh_tasks.append(self._real)

        def done(self) -> bool:
            return False  # always not-done so the loop keeps awaiting

        def cancel(self) -> bool:
            return self._real.cancel()

        def __await__(self) -> Generator[Any]:
            yield from self._real.__await__()
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
    assert conn._pending_drain is None
    asyncio_diagnostics = [
        ctx for ctx in captured if "Task was destroyed" in str(ctx.get("message", ""))
    ]
    if asyncio_diagnostics:
        msgs = [ctx.get("message") for ctx in asyncio_diagnostics]
        raise AssertionError(f"Expected no orphaned-task diagnostics; got {msgs}")


@pytest.mark.asyncio
async def test_close_impl_loop_terminates_when_invalidate_clears_protocol() -> None:
    """The loop terminates after one iteration when the racing ``_invalidate`` clears
    ``_protocol`` (the production callback path) — no infinite spin."""
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
    conn._bound_loop_ref = None

    await conn._close_impl()
    assert pending_a.done()
    assert conn._pending_drain is None
