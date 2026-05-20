"""Pin: ``_connect_impl`` uses a bounded re-snapshot loop (cap=3)
to retire ``_pending_drain``, mirroring ``_close_impl``'s
discipline.

A racing ``_invalidate`` scheduled via ``call_soon_threadsafe``
from the dbapi-sync wrapper's timeout / KI arms can publish a FRESH
``_pending_drain`` task during ``await pending``. The previous
single-shot ``self._pending_drain = None`` at the end of the
retire logic would null the fresh task, orphaning it on the loop
(``"Task was destroyed but it is pending"`` at GC).

Inspection pins on the source plus behavioural pins below that
drive the racing-publish path and the cap-exhausted arm via the
``DqliteConnection.__new__`` hand-seed idiom (see
``test_close_impl_reaps_pending_drain_created_during_await.py``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Generator
from typing import Any

import pytest

from dqliteclient.connection import DqliteConnection


def test_connect_impl_uses_bounded_resnapshot_loop() -> None:
    """``_connect_impl`` must carry the same re-snapshot loop shape
    as ``_close_impl`` so a racing ``_invalidate`` cannot orphan a
    fresh drain task."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    assert "resnapshot_cap = 3" in src, (
        "connect-side re-snapshot loop must use the same cap as close"
    )
    assert "for _attempt in range(resnapshot_cap):" in src
    # Cap-exhausted arm attaches the observer (mirror of close).
    assert "_observe_drain_exception" in src
    assert "stuck.cancel()" in src


def test_connect_impl_cap_exhausted_arm_logs_warning() -> None:
    """The pathological feedback-loop case must surface in
    production logs."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    assert "DqliteConnection._connect_impl: _pending_drain still set after" in src
    assert "feedback loop on connection id" in src


def test_connect_impl_preserves_cancelling_delta_pattern() -> None:
    """The unique connect-side cancelling-delta dance — detect outer
    cancel via ``Task.cancelling()`` counter delta — must survive
    the refactor. Without it, ``connect()`` would silently swallow
    an outer cancel and open a TCP connection the caller intended
    to abort."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    assert "cancelling_before" in src
    assert "cancelling_after" in src
    assert "cancelling_after > cancelling_before" in src


def test_close_impl_and_connect_impl_share_resnapshot_cap_value() -> None:
    """Cross-method symmetry: the cap value matches so a regression
    that bumps one but not the other trips this pin."""
    connect_src = inspect.getsource(DqliteConnection._connect_impl)
    close_src = inspect.getsource(DqliteConnection._close_impl)
    assert "resnapshot_cap = 3" in connect_src
    assert "resnapshot_cap = 3" in close_src


def _seed_connect_conn(address: str = "127.0.0.1:9001") -> DqliteConnection:
    """Hand-seed a ``DqliteConnection`` via ``__new__`` with the
    minimal field set required for ``_connect_impl`` to run through
    the re-snapshot loop and then fail fast on the dial via a
    raising ``dial_func``. The post-loop dial intentionally raises
    so the test stays bounded; the assertions inspect the
    ``_pending_drain`` state AFTER the loop has retired its
    tasks.
    """

    async def _failing_dial(_addr: str) -> Any:
        # Symmetric with the dbapi-sync wrapper's "transport-failure"
        # path: ``open_connection`` re-raises this as-is, the inner
        # ``except OSError`` in ``_connect_impl`` translates it to
        # ``DqliteConnectionError`` which the test catches.
        raise OSError("synthetic dial failure")

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = address
    conn._database = "default"
    conn._timeout = 1.0
    conn._dial_timeout = 1.0
    conn._attempt_timeout = 1.0
    conn._close_timeout = 0.5
    conn._dial_func = _failing_dial
    conn._max_total_rows = None
    conn._max_continuation_frames = None
    conn._trust_server_heartbeat = False
    conn._protocol = None
    conn._db_id = None
    conn._in_transaction = False
    conn._in_use = True  # claimed by connect() before _connect_impl
    conn._closed = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._pool_released = False
    conn._invalidation_cause = None
    conn._pending_drain = None
    conn._bound_loop_ref = None
    conn._connected_flag = [False]
    return conn


@pytest.mark.asyncio
async def test_connect_impl_drains_racing_pending_drain_created_during_await() -> None:
    """Behavioural pin for the re-snapshot loop's race-publish arm.

    Stage a slow drain task A on ``_pending_drain``. Schedule a
    ``call_soon`` callback that, during the loop's ``await pending``
    yield, creates a fresh drain task B and assigns it to
    ``_pending_drain`` — exactly the race the commit message
    documents (``loop.call_soon_threadsafe(dying._invalidate, ...)``
    from the dbapi-sync wrapper).

    The bounded re-snapshot loop must observe B on its second
    iteration and retire it. Without the loop, the single-shot
    ``self._pending_drain = None`` would orphan B on the event loop
    (``"Task was destroyed but it is pending"`` at GC).
    """
    from dqliteclient.exceptions import DqliteConnectionError

    loop = asyncio.get_running_loop()
    conn = _seed_connect_conn()

    async def _slow_drain(_label: str) -> None:
        await asyncio.sleep(0.005)

    pending_a = loop.create_task(_slow_drain("A"))
    pending_b_holder: list[asyncio.Task[None]] = []

    def _race_invalidate_callback() -> None:
        # Fires while _connect_impl is awaiting pending_a — publishes
        # a fresh drain task in the slot, mirroring the production
        # _invalidate path.
        b = loop.create_task(_slow_drain("B"))
        pending_b_holder.append(b)
        conn._pending_drain = b

    conn._pending_drain = pending_a
    loop.call_soon(_race_invalidate_callback)

    captured: list[dict[str, object]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        with pytest.raises(DqliteConnectionError):
            await asyncio.wait_for(conn._connect_impl(), timeout=2.0)
        # Let asyncio surface any orphaned-task diagnostics.
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(prior_handler)

    # Both A and B must be done — the loop reaped both.
    assert pending_a.done()
    assert pending_b_holder, "race callback should have run"
    assert pending_b_holder[0].done(), (
        "fresh _pending_drain task created during await must be reaped "
        "by the re-snapshot loop's second iteration"
    )
    # ``_pending_drain`` is nulled by either the success-break or
    # the cap-exhausted arm — both paths converge on None.
    assert conn._pending_drain is None
    orphan_diagnostics = [
        ctx for ctx in captured if "Task was destroyed" in str(ctx.get("message", ""))
    ]
    if orphan_diagnostics:
        msgs = [ctx.get("message") for ctx in orphan_diagnostics]
        raise AssertionError(f"Expected no orphaned-task diagnostics; got {msgs}")


@pytest.mark.asyncio
async def test_connect_impl_cap_exhausted_arm_warns_and_cancels_stuck(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Behavioural pin for the cap-exhausted arm.

    Pathological feedback loop: a custom awaitable plants a fresh
    not-done task on ``_pending_drain`` after every ``await pending``
    resolves, so each loop iteration sees a fresh non-done task.
    After cap=3 iterations the ``else:`` arm must:

    * cancel the stuck residual task (so the orphan-task diagnostic
      at GC does not fire),
    * attach ``_observe_drain_exception`` to the residual (mirror
      of the close-side discipline),
    * log a WARNING naming the connection id so operators have a
      signal of the pathological feedback loop in production.

    Substring-only pins for these arms would pass a no-op refactor
    that comments-out the cancel / WARN while leaving the constant
    strings present.
    """
    from dqliteclient.exceptions import DqliteConnectionError

    loop = asyncio.get_running_loop()
    conn = _seed_connect_conn()

    fresh_tasks: list[Any] = []
    cancel_calls: list[int] = []
    done_callbacks: list[Any] = []

    class _Adversary:
        """Custom awaitable that plants a fresh _Adversary on
        ``conn._pending_drain`` after every awaited yield — so
        iteration N+1 always sees a non-done pending.

        The connect-side loop calls ``pending.cancel()`` before
        ``await pending`` (mirror of the close-side discipline);
        a real-task adversary would have its sleep cancelled before
        the planting line in ``__await__`` runs, breaking the
        adversary chain after one iteration. We make ``cancel()`` a
        no-op (return False) AND yield once synchronously so the
        planting line always executes regardless of cancel timing.
        """

        def __init__(self) -> None:
            fresh_tasks.append(self)  # track instances for the cap pin

        def done(self) -> bool:
            return False  # always reports not-done so the loop awaits

        def cancel(self) -> bool:
            cancel_calls.append(1)
            return False  # cancel is a no-op so __await__ always completes

        def add_done_callback(self, cb: Any) -> None:
            done_callbacks.append(cb)

        def __await__(self) -> Generator[Any]:
            # Single bare yield so the loop scheduler ticks; on
            # resume, plant the fresh adversary BEFORE returning.
            # Bare yields cannot raise CancelledError unless the
            # loop explicitly delivers it via Task.cancel() — which
            # only applies to the OUTER awaiting task, not our
            # inner awaitable. The outer task's cancelling counter
            # stays at zero, so _connect_impl's
            # ``cancelling_after > cancelling_before`` guard
            # short-circuits the re-raise.
            yield
            # Plant a fresh adversary so the next iteration finds a
            # non-done pending. The slot was nulled by the loop's
            # top-of-iteration ``self._pending_drain = None``; the
            # assignment here re-publishes it (mirror of the racing
            # ``_invalidate`` shape).
            conn._pending_drain = _Adversary()  # type: ignore[assignment]

    conn._pending_drain = _Adversary()  # type: ignore[assignment]

    captured: list[dict[str, object]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    with caplog.at_level(logging.WARNING, logger="dqliteclient.connection"):
        try:
            with pytest.raises(DqliteConnectionError):
                await asyncio.wait_for(conn._connect_impl(), timeout=2.0)
            await asyncio.sleep(0.05)
        finally:
            loop.set_exception_handler(prior_handler)

    # Cap-exhausted WARNING must fire with the connection-id substring.
    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "re-snapshot iterations" in r.getMessage()
        and "_connect_impl" in r.getMessage()
    ]
    assert matching, (
        "Cap-exhausted arm must log a WARNING; without it operators "
        "have no signal of the pathological _invalidate feedback loop"
    )
    rendered = matching[0].getMessage()
    assert f"id={id(conn)}" in rendered, (
        f"WARN must name the connection id for operator correlation: {rendered!r}"
    )
    # The residual stuck task must be cancelled and its observer
    # attached. Both pins guard against the "substring matches a
    # comment" regression scenario the audit calls out.
    assert cancel_calls, (
        "Cap-exhausted arm must call stuck.cancel() so the residual task does not orphan at GC"
    )
    assert done_callbacks, (
        "Cap-exhausted arm must attach _observe_drain_exception so "
        "the stuck task's exception is observed (mirror of _close_impl)"
    )
    # Slot was cleared so no orphan-task diagnostic fires at GC.
    assert conn._pending_drain is None
    orphan_diagnostics = [
        ctx for ctx in captured if "Task was destroyed" in str(ctx.get("message", ""))
    ]
    if orphan_diagnostics:
        msgs = [ctx.get("message") for ctx in orphan_diagnostics]
        raise AssertionError(f"Expected no orphaned-task diagnostics; got {msgs}")
