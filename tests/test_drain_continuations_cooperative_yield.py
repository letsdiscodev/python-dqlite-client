"""_drain_continuations yields each iteration so a fast-burst server can't monopolise the loop.

StreamReader.read* returns synchronously when bytes are already buffered, so the drain needs an
explicit ``await asyncio.sleep(0)`` per iteration or sibling coroutines starve for the whole drain.
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
    """Synchronously-decoded continuation frames must still yield to siblings each iteration."""
    proto = _make_protocol()

    # 200 prefetched continuation frames + a final has_more=False frame.
    cont_frames = [
        RowsResponse(column_names=["x"], rows=[[i]], has_more=True) for i in range(1, 201)
    ]
    cont_frames.append(RowsResponse(column_names=["x"], rows=[[201]], has_more=False))
    cont_iter = iter(cont_frames)

    async def fake_read_continuation(deadline: float) -> RowsResponse:
        # No await: mirrors the StreamReader fast path with the next frame already buffered.
        return next(cont_iter)

    proto._read_continuation = fake_read_continuation  # type: ignore[assignment]

    sibling_ran = 0

    async def sibling() -> None:
        nonlocal sibling_ran
        while True:
            await asyncio.sleep(0)
            sibling_ran += 1

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)

    sibling_task = asyncio.create_task(sibling())
    try:
        # Schedule the sibling before the drain claims the loop.
        await asyncio.sleep(0)
        baseline = sibling_ran

        rows, _types = await proto._drain_continuations(initial, deadline=999999.0)

        during = sibling_ran - baseline
    finally:
        sibling_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sibling_task

    assert len(rows) == 202, "all frames must be drained"
    # >= 50 (vs 0 under the prior code) guards against a future yield-every-Nth refactor.
    assert during >= 50, (
        f"sibling ran only {during} times while drain processed 201 "
        "prefetched frames; cooperative yield missing"
    )
