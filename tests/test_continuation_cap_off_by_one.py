"""max_continuation_frames is an exact upper bound: cap N allows at most N
decoded frames (initial + continuations). The ``>=`` predicate avoids N+1 slop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import RowsResponse


def _make_protocol(max_continuation_frames: int = 2) -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(
        reader,
        writer,
        timeout=5.0,
        max_continuation_frames=max_continuation_frames,
    )


@pytest.mark.asyncio
async def test_drain_continuations_cap_is_inclusive() -> None:
    """Cap=2 allows initial + one continuation; a third frame must raise."""
    proto = _make_protocol(max_continuation_frames=2)
    cont_frames = iter(
        [
            RowsResponse(column_names=["x"], rows=[[1]], has_more=True),
            RowsResponse(column_names=["x"], rows=[[2]], has_more=False),
        ]
    )

    async def fake_read_cont(deadline: float) -> RowsResponse:
        return next(cont_frames)

    proto._read_continuation = fake_read_cont

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)
    with pytest.raises(ProtocolError, match="max_continuation_frames"):
        await proto._drain_continuations(initial, deadline=0.0)


@pytest.mark.asyncio
async def test_drain_continuations_cap_at_exact_limit_accepted() -> None:
    """Cap=2 with exactly 2 decoded frames must succeed."""
    proto = _make_protocol(max_continuation_frames=2)
    cont_frames = iter([RowsResponse(column_names=["x"], rows=[[1]], has_more=False)])

    async def fake_read_cont(deadline: float) -> RowsResponse:
        return next(cont_frames)

    proto._read_continuation = fake_read_cont

    initial = RowsResponse(column_names=["x"], rows=[[0]], has_more=True)
    rows, _types = await proto._drain_continuations(initial, deadline=0.0)
    assert rows == [[0], [1]]
