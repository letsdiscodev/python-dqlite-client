"""Pin: ``ConnectionPool.close()`` must keep ``_finalizer.detach()`` inside
the ``try:`` frame so a ``BaseException`` between flag publication and
the drain body cannot leave ``_close_done`` constructed-but-never-set.

Threat model: a synthetic ``BaseException`` (KeyboardInterrupt /
SystemExit / signal-driven cancel) delivered between
``self._closed = True`` and the first awaited line of the close body
would skip the ``try`` frame entirely. The ``finally`` arm —
``self._close_done.set()`` — would never run. A second caller that
observes ``_closed=True`` would then await ``_close_done`` forever,
violating the pool's "Idempotent and concurrent-caller-safe" docstring
contract.

Fix: every line that runs after the flag flip lives inside the
``try`` frame. The finalizer detach is a synchronous bytecode-local
operation, so the cost is zero — the only behavioural change is that
``_close_done.set()`` is now reachable even if the synchronous body
raises.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_finalizer_detach_in_close_lives_inside_try_frame() -> None:
    """A synthetic ``BaseException`` raised by the finalizer's
    ``detach`` call must still leave ``_close_done`` set, so a second
    caller's early-return arm does not hang."""

    pool = ConnectionPool(addresses=["10.0.0.1:9001"], min_size=0, max_size=1)
    pool._closed_event = asyncio.Event()
    sentinel = SystemExit("synthetic")

    class _BoomFinalizer:
        def detach(self) -> None:
            raise sentinel

    pool._finalizer = _BoomFinalizer()  # type: ignore[assignment]

    with pytest.raises(SystemExit) as excinfo:
        await pool.close()

    assert excinfo.value is sentinel
    assert pool._closed is True
    assert pool._close_done is not None
    assert pool._close_done.is_set(), (
        "close()'s try/finally must set _close_done even when the "
        "body raises BaseException, so the second-caller early-return "
        "arm at the top of close() does not hang on _close_done.wait()."
    )


@pytest.mark.asyncio
async def test_second_caller_does_not_hang_when_first_caller_raised() -> None:
    """End-to-end: first ``close()`` caller raises mid-body; second
    caller awaiting on the same pool must still return."""

    pool = ConnectionPool(addresses=["10.0.0.1:9001"], min_size=0, max_size=1)
    pool._closed_event = asyncio.Event()

    class _BoomFinalizer:
        def detach(self) -> None:
            raise SystemExit("synthetic")

    pool._finalizer = _BoomFinalizer()  # type: ignore[assignment]

    # First caller raises.
    with pytest.raises(SystemExit):
        await pool.close()

    # Second caller should not hang. Wrap in a tight timeout so a
    # regression turns this test into a fast failure rather than a
    # session-wide hang.
    await asyncio.wait_for(pool.close(), timeout=2.0)
