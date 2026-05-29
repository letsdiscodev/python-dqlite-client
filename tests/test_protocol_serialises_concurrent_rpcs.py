"""Pin: ``DqliteProtocol`` serialises concurrent wire-touching RPCs
behind an internal ``asyncio.Lock``. The dqlite server does not
support concurrent requests on one connection; without the lock,
interleaved writes/reads poison the codec several round-trips later.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import LeaderResponse

pytestmark = pytest.mark.asyncio


async def test_protocol_has_asyncio_lock_attribute() -> None:
    """``DqliteProtocol`` exposes an ``asyncio.Lock`` ``_lock``."""
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    proto = DqliteProtocol(reader, writer)
    assert isinstance(proto._lock, asyncio.Lock)


async def test_concurrent_get_leader_calls_serialise() -> None:
    """Two concurrent ``get_leader()`` calls run sequentially; each
    read pulls its own response from the shared decoder."""
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()

    # Distinct payloads so each task can tell which response it got.
    resp_a = LeaderResponse(node_id=1, address="10.0.0.1:9001").encode()
    resp_b = LeaderResponse(node_id=2, address="10.0.0.2:9001").encode()

    payloads = [resp_a, resp_b]
    proto = DqliteProtocol(reader, writer)

    start_a = asyncio.Event()
    release_a = asyncio.Event()
    started_a = asyncio.Event()
    started_b = asyncio.Event()

    call_order: list[str] = []
    read_order: list[str] = []

    async def fake_read(n: int) -> bytes:
        # Hold both reads until both tasks have queued, so the lock
        # has a chance to enforce serialisation.
        if "A_first_read" not in call_order:
            call_order.append("A_first_read")
            await start_a.wait()
        await release_a.wait()
        if payloads:
            data = payloads.pop(0)
            read_order.append(data.hex()[:16])
            return data
        return b""

    reader.read = AsyncMock(side_effect=fake_read)

    async def task_a() -> tuple[int, str]:
        started_a.set()
        try:
            return await proto.get_leader()
        finally:
            call_order.append("A_done")

    async def task_b() -> tuple[int, str]:
        started_b.set()
        try:
            return await proto.get_leader()
        finally:
            call_order.append("B_done")

    fut_a = asyncio.create_task(task_a())
    await started_a.wait()
    # Give A a chance to acquire the lock and send.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    fut_b = asyncio.create_task(task_b())
    await started_b.wait()
    start_a.set()
    release_a.set()

    result_a, result_b = await asyncio.gather(fut_a, fut_b)

    assert call_order.index("A_done") < call_order.index("B_done"), (
        f"DqliteProtocol must serialise RPCs; observed order {call_order!r}"
    )
    assert result_a == (1, "10.0.0.1:9001")
    assert result_b == (2, "10.0.0.2:9001")
