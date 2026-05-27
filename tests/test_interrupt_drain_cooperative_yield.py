"""Pin: ``_interrupt``'s drain-until-EMPTY loop yields cooperatively
each iteration so a pre-buffered trailing ROWS / RESULT stream cannot
monopolise the event loop.

asyncio's ``StreamReader.read*`` returns synchronously when the
requested bytes are already buffered, so ``await
self._read_response(...)`` inside the drain loop does NOT yield to
the loop scheduler on the fast-burst path. Without an explicit
``await asyncio.sleep(0)`` at the bottom of the loop, an INTERRUPT
against a pre-buffered stream pins the loop for the full drain —
direct sibling of the already-fixed ``_drain_continuations``
prefetched-frames hazard.
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
    """Many pre-buffered ROWS frames followed by the terminal EMPTY
    must not starve sibling coroutines. ``_interrupt`` must yield
    to the loop scheduler at least once per iteration even when
    every ``_read_response`` returns without parking on the wire.
    """
    proto = _make_protocol()

    # 200 pre-buffered ROWS frames + the EMPTY ack at the end.
    responses: list[RowsResponse | EmptyResponse] = [
        RowsResponse(column_names=["x"], rows=[[i]], has_more=True) for i in range(1, 201)
    ]
    responses.append(EmptyResponse())
    responses_iter = iter(responses)

    async def fake_read_response(
        deadline: float | None = None, allow_trailing: bool = False
    ) -> object:
        # No await: synchronous return mirrors the StreamReader fast
        # path where the next frame is already buffered.
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
        # Yield once so the sibling task is scheduled before the
        # drain claims the loop.
        await asyncio.sleep(0)
        baseline = sibling_ran

        await proto._interrupt(db_id=0)

        during = sibling_ran - baseline
    finally:
        sibling_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sibling_task

    # The drain ran 200 ROWS iterations + 1 EMPTY ack. Under the
    # cooperative-yield fix the sibling should pick up at least a
    # non-trivial fraction of those. Pin at >= 50: any value > 0
    # proves the drain yields at least sometimes; the higher bound
    # guards against a future refactor that yields only every Nth
    # iteration with a large N. Under the prior code this was zero.
    assert during >= 50, (
        f"sibling ran only {during} times while _interrupt drained "
        "200+ prefetched frames; cooperative yield missing"
    )
