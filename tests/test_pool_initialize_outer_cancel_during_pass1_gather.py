"""Pin: ``ConnectionPool.initialize``'s pass-1 cleanup ``gather`` is
itself robust to an outer cancel — pass-2 (close already-completed
tasks' connections) still runs even if pass-1's gather raises
CancelledError.

Pre-fix the finally walked:

1. cancel-pending → ``await asyncio.gather(*cancelled_pending,
   return_exceptions=True)``;
2. for completed-but-not-cancelled tasks, extract result and append
   to ``unqueued_survivors`` so they get closed at the end of the
   finally.

If an outer ``BaseException`` (e.g. ``asyncio.timeout`` firing
during pass-1's gather) propagated into pass-1's gather, the gather
re-raised CancelledError, the function returned without running pass-2,
and connections produced by tasks that completed JUST BEFORE being
cancelled were orphaned (transports leaked → ResourceWarning at GC).

This test wires:

- 3 ``_create_connection`` tasks: the first two return a fake conn
  immediately; the third blocks forever.
- The outer cleanup gather is intercepted: when pass-1's
  ``asyncio.gather`` is awaited, raise CancelledError before it
  resolves the not-yet-done third task. The two already-completed
  tasks have their ``_result`` slots set.

Pin: the two completed-task connections are observed-closed (their
``close()`` was awaited), even though pass-1's gather raised.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_outer_cancel_during_pass1_gather_still_closes_completed_conns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    completed_conns: list[_FakeConn] = []

    block_event = asyncio.Event()
    create_call = {"n": 0}

    async def fake_create_connection() -> Any:
        n = create_call["n"]
        create_call["n"] += 1
        if n < 2:
            c = _FakeConn()
            completed_conns.append(c)
            return c
        # The third task blocks until cancelled.
        await block_event.wait()
        # Should not reach.
        raise AssertionError("blocking task resumed unexpectedly")

    pool._create_connection = fake_create_connection

    # Force the outer ``asyncio.gather`` (called from
    # initialize) to raise CancelledError so the function exits via
    # finally without running the post-gather success path. We then
    # additionally make the pass-1 gather inside the finally raise
    # CancelledError to mimic an outer ``asyncio.timeout`` firing
    # during pass-1's cleanup gather.
    real_gather = asyncio.gather
    gather_calls = {"n": 0}

    async def patched_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["n"] += 1
        if gather_calls["n"] == 1:
            # First gather call is the success-path gather inside
            # initialize. Cancel one of the running tasks so gather
            # raises and we go into the finally with one task pending
            # and the other two completed.
            #
            # Simpler approach: forward to the real gather, but
            # cancel pool._create_connection-bound tasks first so the
            # real gather collects two completes + one cancelled.
            # Easiest implementation: schedule a cancel of the third
            # task immediately and let the real gather run, then
            # raise CancelledError at the gather frame (rather than
            # bubbling each task's outcome).
            for t in args:
                if isinstance(t, asyncio.Task) and not t.done():
                    # Snapshot tasks; we cancel only the still-running
                    # ones at the outer loop tick.
                    pass
            # Yield once so the first two creates land their results.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Now raise CancelledError out of the gather frame to
            # simulate outer cancel.
            raise asyncio.CancelledError("synthetic-outer-cancel-during-gather")
        if gather_calls["n"] == 2:
            # The pass-1 cleanup gather. Mimic outer
            # asyncio.timeout firing inside this gather.
            #
            # Cancel each pending task so they at least get awaited
            # by us before we synthesise the error.
            for t in args:
                if isinstance(t, asyncio.Task) and not t.done():
                    t.cancel()
                    with contextlib.suppress(BaseException):
                        await t
            raise asyncio.CancelledError("synthetic-outer-cancel-during-pass1")
        return await real_gather(*args, **kwargs)

    monkeypatch.setattr("asyncio.gather", patched_gather)

    with pytest.raises(asyncio.CancelledError):
        await pool.initialize()

    # Pin: both completed-task connections were closed despite
    # pass-1's gather raising CancelledError.
    assert len(completed_conns) == 2, (
        f"test wired wrong: expected 2 completed conns, got {len(completed_conns)}"
    )
    for c in completed_conns:
        assert c.closed, (
            "completed-task connection was orphaned when outer cancel hit "
            "pass-1's gather (pass-2 close walk was skipped)"
        )

    # Tidy up the pool side.
    pool._create_connection = AsyncMock(side_effect=AssertionError)
