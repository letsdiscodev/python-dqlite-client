"""Pin: ``pool.close()`` does not leak un-closed idle connections when
an outer cancel lands mid-drain across multiple queued connections.

The sibling ``test_pool_drain_shielded_against_outer_cancel.py`` pins
the per-connection shield: a single ``conn.close()`` started before
the cancel completes cleanly. This test pins the queue-as-a-whole
contract: an outer ``asyncio.timeout(pool.close())`` that fires after
N of M connections have started closing must NOT leave the remaining
M-N queued connections orphaned.

Acceptable post-cancel states:

(a) drain finished (cancel landed past the loop) — every connection
    has had ``close()`` called and completed.
(b) drain partially complete and bailed on cancel — but every
    connection that was *started* completed under the shield.

NOT acceptable:

(c) some connections never had ``close()`` called and are leaked in
    the queue.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_drain_does_not_orphan_remaining_queued_connections() -> None:
    started: list[int] = []
    completed: list[int] = []

    class FakeConn:
        def __init__(self, conn_id: int) -> None:
            self._id = conn_id
            self._address = f"h{conn_id}:9"

        async def close(self) -> None:
            started.append(self._id)
            await asyncio.sleep(0.05)
            completed.append(self._id)

    fakes = [FakeConn(i) for i in range(10)]

    pool = ConnectionPool.__new__(ConnectionPool)
    pool._pool = asyncio.Queue()
    for f in fakes:
        pool._pool.put_nowait(f)  # type: ignore[arg-type]
    pool._size = len(fakes)
    pool._max_size = len(fakes)
    pool._lock = asyncio.Lock()
    pool._closed = False
    pool._closed_event = None

    # Race the outer cancel against the drain. 0.12s lets the loop
    # start the second connection (each close is 0.05s), then the
    # cancel lands mid-iteration. If the loop bails without taking
    # extra connections, those connections never get close() called.
    drain_task = asyncio.create_task(pool._drain_idle())
    await asyncio.sleep(0.12)
    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    # Yield long enough for any in-flight shielded closes to land.
    await asyncio.sleep(0.5)

    # Contract: every close() that was STARTED must have COMPLETED.
    assert set(started) == set(completed), (
        f"some close() started but did not complete: "
        f"started={sorted(started)} completed={sorted(completed)}"
    )

    # Contract: the queue MUST NOT still hold connections that were
    # never given a chance to close. Either the drain completed
    # (state a) or it bailed on cancel — but the bail must not leave
    # never-touched FakeConns sitting in the queue indefinitely.
    # The current implementation iterates with ``while not empty``,
    # so when cancel propagates the next iteration's get_nowait is
    # skipped. Pin this with the orphan check: any connection still
    # sitting in the queue after cancel is "never had close called".
    # The drain's finally clause runs ``_drain_remaining_after_cancel``
    # under shield, which sweeps any remaining queue entries and calls
    # close() on each. Both the main-loop closes and the cleanup-pass
    # closes append to ``started``/``completed``, so the contract is:
    # every original FakeConn was closed, none was orphaned in the
    # queue.
    assert pool._pool.empty(), (
        "pool queue should be drained after close cancel — "
        "_drain_remaining_after_cancel must sweep what the main loop did not"
    )
    assert sorted(completed) == list(range(10)), (
        f"every queued connection must have completed close(): completed={sorted(completed)}"
    )
