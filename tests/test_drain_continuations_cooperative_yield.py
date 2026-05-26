"""Pin: ``_drain_continuations`` yields cooperatively each iteration so
a fast-burst server (many small frames already buffered in the
``StreamReader``) cannot monopolise the event loop for the full drain.

asyncio's ``StreamReader.read*`` returns immediately when the requested
bytes are already buffered, without scheduling a loop tick. So
``await self._read_continuation(...)`` inside the drain loop does NOT
yield to other coroutines on the fast-burst path. Without an explicit
``await asyncio.sleep(0)`` at the bottom of the loop, a sibling
coroutine scheduled before the drain is starved for the duration of
the entire drain — heartbeat probes, pool acquirers, sibling RPCs all
freeze.

This test pins the cooperative-yield invariant directly: drain a
prefetched-frame stream and assert a sibling coroutine gets at least
one slice of loop time before the drain finishes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import RowsResponse


def _make_protocol() -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=5.0)


@pytest.mark.asyncio
async def test_drain_continuations_yields_between_prefetched_frames() -> None:
    """Many continuation frames already decoded synchronously must not
    starve sibling coroutines. ``_drain_continuations`` must yield to
    the loop scheduler at least once per iteration even when every
    ``_read_continuation`` returns without parking on the wire.
    """
    proto = _make_protocol()

    # 200 prefetched continuation frames + a final has_more=False frame.
    cont_frames = [
        RowsResponse(column_names=["x"], rows=[[i]], has_more=True) for i in range(1, 201)
    ]
    cont_frames.append(RowsResponse(column_names=["x"], rows=[[201]], has_more=False))
    cont_iter = iter(cont_frames)

    async def fake_read_continuation(deadline: float) -> RowsResponse:
        # No await: synchronous return mirrors the StreamReader fast
        # path where the next frame is already buffered.
        return next(cont_iter)

    proto._read_continuation = fake_read_continuation  # type: ignore[assignment]

    sibling_ran = 0

    async def sibling() -> None:
        nonlocal sibling_ran
        # Count how many times the sibling gets to run while the drain
        # is executing. A non-yielding drain pins the loop and the
        # sibling never gets a turn until the drain returns.
        while True:
            await asyncio.sleep(0)
            sibling_ran += 1

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)

    sibling_task = asyncio.create_task(sibling())
    try:
        # Yield once so the sibling task is scheduled before the drain
        # claims the loop.
        await asyncio.sleep(0)
        baseline = sibling_ran

        rows, _types = await proto._drain_continuations(initial, deadline=999999.0)

        during = sibling_ran - baseline
    finally:
        sibling_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sibling_task

    assert len(rows) == 202, "all frames must be drained"
    # The drain ran 201 iterations; under the cooperative-yield fix the
    # sibling should pick up at least a non-trivial fraction of those.
    # Pin at >= 50: any value greater than zero proves the drain yields
    # at least sometimes; the higher bound guards against a future
    # refactor that yields only on every Nth iteration with a very
    # large N. Under the prior code this was zero.
    assert during >= 50, (
        f"sibling ran only {during} times while drain processed 201 "
        "prefetched frames; cooperative yield missing"
    )
