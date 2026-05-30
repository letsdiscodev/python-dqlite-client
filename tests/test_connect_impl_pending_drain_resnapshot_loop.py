"""``_connect_impl`` retires ``_pending_drain`` via a bounded re-snapshot loop
(cap=3), mirroring ``_close_impl``, so a racing ``_invalidate`` that publishes a
fresh drain task during ``await pending`` cannot orphan it (the old single-shot
null would leave it pending at GC)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from typing import Any

import pytest

from dqliteclient.connection import DqliteConnection


def _seed_connect_conn(address: str = "127.0.0.1:9001") -> DqliteConnection:
    """Hand-seed a ``DqliteConnection`` via ``__new__`` that runs the re-snapshot
    loop then fails fast on the dial (keeps the test bounded)."""

    async def _failing_dial(_addr: str) -> Any:
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
    """A fresh drain task B published during ``await pending`` (the race the commit
    documents) must be observed and retired on the loop's second iteration."""
    from dqliteclient.exceptions import DqliteConnectionError

    loop = asyncio.get_running_loop()
    conn = _seed_connect_conn()

    async def _slow_drain(_label: str) -> None:
        await asyncio.sleep(0.005)

    pending_a = loop.create_task(_slow_drain("A"))
    pending_b_holder: list[asyncio.Task[None]] = []

    def _race_invalidate_callback() -> None:
        # Fires while awaiting pending_a; publishes a fresh drain task, mirroring
        # the production _invalidate path.
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

    assert pending_a.done()
    assert pending_b_holder, "race callback should have run"
    assert pending_b_holder[0].done(), (
        "fresh _pending_drain task created during await must be reaped "
        "by the re-snapshot loop's second iteration"
    )
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
    """After cap=3 iterations of a fresh-task-planting adversary, the else arm
    must cancel the stuck task, attach the observer, and WARN with the conn id."""
    from dqliteclient.exceptions import DqliteConnectionError

    loop = asyncio.get_running_loop()
    conn = _seed_connect_conn()

    fresh_tasks: list[Any] = []
    cancel_calls: list[int] = []
    done_callbacks: list[Any] = []

    class _Adversary:
        """Plants a fresh _Adversary after every awaited yield. cancel() is a
        no-op so the planting line runs despite the loop's pre-await cancel()."""

        def __init__(self) -> None:
            fresh_tasks.append(self)

        def done(self) -> bool:
            return False  # never done, so the loop always awaits

        def cancel(self) -> bool:
            cancel_calls.append(1)
            return False  # no-op so __await__ always completes

        def add_done_callback(self, cb: Any) -> None:
            done_callbacks.append(cb)

        def __await__(self) -> Generator[Any]:
            # Bare yield: the outer task's cancelling counter stays at zero so
            # _connect_impl's cancelling-delta guard does not re-raise.
            yield
            # Re-publish a fresh adversary (mirror of the racing _invalidate).
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
    assert cancel_calls, (
        "Cap-exhausted arm must call stuck.cancel() so the residual task does not orphan at GC"
    )
    assert done_callbacks, (
        "Cap-exhausted arm must attach _observe_drain_exception so "
        "the stuck task's exception is observed (mirror of _close_impl)"
    )
    assert conn._pending_drain is None
    orphan_diagnostics = [
        ctx for ctx in captured if "Task was destroyed" in str(ctx.get("message", ""))
    ]
    if orphan_diagnostics:
        msgs = [ctx.get("message") for ctx in orphan_diagnostics]
        raise AssertionError(f"Expected no orphaned-task diagnostics; got {msgs}")
