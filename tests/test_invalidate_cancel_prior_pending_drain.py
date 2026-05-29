"""Pin ``_invalidate``'s cancel-and-detach guard: a second call cancels
the prior bounded-drain task before overwriting ``_pending_drain`` (else
it orphans, recreating the "Task was destroyed" warning). Also pins the
closed-loop ``create_task`` fallback (preserve cause, drain=None) and the
``_invalidation_cause`` clear on ``_close_impl``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from dqliteclient.connection import DqliteConnection


def _make_conn_with_protocol() -> DqliteConnection:
    """DqliteConnection skeleton to drive ``_invalidate`` without a wire connection."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._protocol = None
    conn._db_id = None
    conn._pending_drain = None
    conn._invalidation_cause = None
    conn._in_use = False
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._closed = False
    conn._close_timeout = 0.05
    conn._address = "test:9001"
    conn._bound_loop_ref = None
    conn._pool_released = False
    return conn


class _FakeProtocol:
    """Minimal DqliteProtocol stand-in: ``close()`` and ``wait_closed()``."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def wait_closed(self) -> None:
        # Stay pending long enough for a sibling _invalidate to observe the drain.
        await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_second_invalidate_cancels_prior_pending_drain() -> None:
    conn = _make_conn_with_protocol()

    # First invalidate schedules a bounded-drain task.
    conn._protocol = _FakeProtocol()  # type: ignore[assignment]
    conn._invalidate()
    first_task = conn._pending_drain
    assert first_task is not None
    assert not first_task.done()

    # Re-set the protocol so the second _invalidate's branch runs.
    conn._protocol = _FakeProtocol()  # type: ignore[assignment]
    conn._invalidate()
    second_task = conn._pending_drain

    assert second_task is not None
    assert second_task is not first_task
    # cancel() flips to ``cancelling`` until the task observes it at its
    # next await, so pump the loop below to let the cancellation land.
    assert first_task.cancelling() > 0 or first_task.cancelled() or first_task.done()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert first_task.cancelled() or first_task.done()


@pytest.mark.asyncio
async def test_invalidate_with_closed_loop_preserves_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A closed-loop ``create_task`` must preserve the original cause;
    the guard DEBUG-logs and continues with ``_pending_drain = None``."""
    conn = _make_conn_with_protocol()
    conn._protocol = _FakeProtocol()  # type: ignore[assignment]

    # Patch create_task to simulate the closed-loop shape during dispose.
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

    assert conn._pending_drain is None
    assert isinstance(conn._invalidation_cause, ValueError)
    assert any("asyncio.ensure_future" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_close_impl_clears_invalidation_cause() -> None:
    """``_close_impl`` clears ``_invalidation_cause`` so the cached
    exception does not pin frame globals/locals across close/reconnect."""
    conn = _make_conn_with_protocol()
    conn._invalidation_cause = ValueError("prior failure")
    assert conn._invalidation_cause is not None

    await conn._close_impl()

    assert conn._invalidation_cause is None
