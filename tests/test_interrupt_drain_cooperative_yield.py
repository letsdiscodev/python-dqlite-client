"""Pin: ``_interrupt``'s drain-until-EMPTY loop yields cooperatively
each iteration (StreamReader.read* returns synchronously when bytes are
already buffered, so a pre-buffered stream would otherwise pin the loop).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import EmptyResponse, RowsResponse


def _make_protocol() -> DqliteProtocol:
    reader = MagicMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=5.0)


@pytest.mark.asyncio
async def test_interrupt_drain_yields_between_prefetched_frames() -> None:
    """Many pre-buffered ROWS frames must not starve siblings: ``_interrupt``
    yields at least once per iteration even when reads never park."""
    proto = _make_protocol()

    responses: list[RowsResponse | EmptyResponse] = [
        RowsResponse(column_names=["x"], rows=[[i]], has_more=True) for i in range(1, 201)
    ]
    responses.append(EmptyResponse())
    responses_iter = iter(responses)

    async def fake_read_response(
        deadline: float | None = None, allow_trailing: bool = False
    ) -> object:
        # Synchronous return mirrors the StreamReader fast path (frame already buffered).
        return next(responses_iter)

    async def fake_send(frame: bytes) -> None:
        return None

    proto._read_response = fake_read_response  # type: ignore[assignment]
    proto._send = fake_send

    sibling_ran = 0

    async def sibling() -> None:
        nonlocal sibling_ran
        while True:
            await asyncio.sleep(0)
            sibling_ran += 1

    sibling_task = asyncio.create_task(sibling())
    try:
        # Schedule the sibling before the drain claims the loop.
        await asyncio.sleep(0)
        baseline = sibling_ran

        await proto._interrupt(db_id=0)

        during = sibling_ran - baseline
    finally:
        sibling_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sibling_task

    # >= 50 (was 0 pre-fix): guards against a refactor that yields only every Nth iteration.
    assert during >= 50, (
        f"sibling ran only {during} times while _interrupt drained "
        "200+ prefetched frames; cooperative yield missing"
    )
