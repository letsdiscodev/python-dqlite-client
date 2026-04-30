"""Pin: cycle 22's cancel-and-detach guard on
``DqliteConnection._invalidate``.

A second ``_invalidate`` call while a prior bounded-drain task is
in-flight must cancel the prior task before overwriting
``self._pending_drain`` with the new one. Without the cancel,
the first task is orphaned (still running on the loop, no longer
reachable from ``self``), and ``close()`` awaits only the second
— recreating the exact "Task was destroyed but it is pending"
warning the drain mechanism was added to suppress.

Also pins the cycle 22 ``RuntimeError("Event loop is closed")``
guard around ``loop.create_task``: when the loop is dead the
fallback sets ``_pending_drain = None`` and preserves the
original cancel/cause instead of replacing it with a bare
``RuntimeError``.

And pins the ``_invalidation_cause`` clear on ``_close_impl``
(cycle 22) so the cached exception does not pin frame
globals/locals across close → reconnect cycles.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from dqliteclient.connection import DqliteConnection


def _make_conn_with_protocol() -> DqliteConnection:
    """Build a DqliteConnection skeleton sufficient to drive
    ``_invalidate`` end-to-end without a real wire connection."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._protocol = None
    conn._db_id = None
    conn._pending_drain = None
    conn._invalidation_cause = None
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._closed = False  # type: ignore[attr-defined]
    conn._close_timeout = 0.05
    conn._address = "test:9001"
    conn._bound_loop = None
    conn._pool_released = False
    return conn


class _FakeProtocol:
    """Minimal stand-in for DqliteProtocol covering the slots
    ``_invalidate`` reads (``close()``, ``wait_closed()``)."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def wait_closed(self) -> None:
        # Yield once so the bounded drain has a chance to be
        # observed in pending state by a sibling _invalidate.
        await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_second_invalidate_cancels_prior_pending_drain() -> None:
    conn = _make_conn_with_protocol()

    # First invalidate: scheduling a bounded-drain task.
    conn._protocol = _FakeProtocol()  # type: ignore[assignment]
    conn._invalidate()
    first_task = conn._pending_drain
    assert first_task is not None
    assert not first_task.done()

    # Re-set the protocol so the second _invalidate's
    # ``if self._protocol is not None:`` branch runs.
    conn._protocol = _FakeProtocol()  # type: ignore[assignment]
    conn._invalidate()
    second_task = conn._pending_drain

    assert second_task is not None
    assert second_task is not first_task
    # The cycle 22 contract: prior task has cancel() scheduled
    # before the slot is overwritten. ``cancel()`` flips the task
    # to ``cancelling`` (not yet ``cancelled``) until the task
    # observes the CancelledError at its next await — pump the
    # loop so the cancellation lands.
    assert first_task.cancelling() > 0 or first_task.cancelled() or first_task.done()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert first_task.cancelled() or first_task.done()


@pytest.mark.asyncio
async def test_invalidate_with_closed_loop_preserves_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``loop.create_task`` raising
    ``RuntimeError("Event loop is closed")`` must NOT replace the
    original cancel/cause; cycle 22 added the try/except guard
    that DEBUG-logs and continues with ``_pending_drain = None``."""
    conn = _make_conn_with_protocol()
    conn._protocol = _FakeProtocol()  # type: ignore[assignment]

    # Patch the running loop's create_task to simulate the
    # closed-loop shape during dispose.
    loop = asyncio.get_running_loop()
    real_create_task = loop.create_task

    def _raise_loop_closed(coro: object, **kwargs: object) -> object:
        # Close the coroutine to suppress "never awaited" warnings.
        coro.close()  # type: ignore[attr-defined]
        raise RuntimeError("Event loop is closed")

    loop.create_task = _raise_loop_closed  # type: ignore[assignment]
    try:
        caplog.set_level(logging.DEBUG, logger="dqliteclient.connection")
        conn._invalidate(cause=ValueError("original cause"))
    finally:
        loop.create_task = real_create_task

    # ``_pending_drain`` falls back to None instead of leaking the
    # un-scheduled coroutine; the original cause is preserved on
    # ``_invalidation_cause`` for downstream chaining.
    assert conn._pending_drain is None
    assert isinstance(conn._invalidation_cause, ValueError)
    assert any("loop.create_task" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_close_impl_clears_invalidation_cause() -> None:
    """Cycle 22 added ``self._invalidation_cause = None`` on
    ``_close_impl`` so the cached exception does NOT pin
    frame globals/locals across close → reconnect cycles. A
    regression that drops the line re-introduces the
    traceback-pin defect."""
    conn = _make_conn_with_protocol()
    # Simulate a prior invalidation having stored a cause.
    conn._invalidation_cause = ValueError("prior failure")
    assert conn._invalidation_cause is not None

    await conn._close_impl()

    assert conn._invalidation_cause is None
