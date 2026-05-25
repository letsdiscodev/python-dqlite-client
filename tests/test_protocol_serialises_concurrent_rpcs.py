"""Pin: ``DqliteProtocol`` serialises concurrent wire-touching RPCs
behind an internal ``asyncio.Lock``, matching go-dqlite's
``Protocol.mu sync.Mutex`` (``internal/protocol/protocol.go:15-30``).

The dqlite server does not support concurrent requests on a single
connection. Without serialisation, two coroutines awaiting different
RPCs on the same protocol instance interleave their writes on the
shared writer and their reads on the shared decoder, surfacing as a
malformed-frame ``ProtocolError`` or codec poisoning several round-
trips later with no breadcrumb pointing at the concurrency violation.

In-tree callers route through ``DqliteConnection``'s ``_in_use`` flag
which guards the same race higher up the stack. Third-party callers
that import ``DqliteProtocol`` directly were not covered. The
protocol-layer lock closes that gap.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import LeaderResponse

pytestmark = pytest.mark.asyncio


async def test_protocol_has_asyncio_lock_attribute() -> None:
    """Pin: ``DqliteProtocol`` exposes a ``_lock`` attribute of type
    ``asyncio.Lock`` after construction.
    """
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    proto = DqliteProtocol(reader, writer)
    assert isinstance(proto._lock, asyncio.Lock)


async def test_concurrent_get_leader_calls_serialise() -> None:
    """Two concurrent ``get_leader()`` calls must produce sequential
    request/response pairs on the wire — no interleaving. With the
    lock in place, the second task is queued behind the first
    regardless of arrival order; reads from the shared decoder are
    therefore matched to the correct request.
    """
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()

    # Build two distinct LeaderResponse payloads so each task knows
    # which response it should see if calls are serialised correctly.
    resp_a = LeaderResponse(node_id=1, address="10.0.0.1:9001").encode()
    resp_b = LeaderResponse(node_id=2, address="10.0.0.2:9001").encode()

    # Order the bytes such that A's response is read first, B's second.
    # If the lock holds, task A's await of _read_response will consume
    # resp_a and task B's await will consume resp_b. Without the lock
    # they could interleave so we'd get either order or codec poison.
    payloads = [resp_a, resp_b]
    # Block reads on an event to inject deterministic ordering — A
    # claims the lock, then B queues, then we let the reads complete.
    proto = DqliteProtocol(reader, writer)

    start_a = asyncio.Event()
    release_a = asyncio.Event()
    started_a = asyncio.Event()
    started_b = asyncio.Event()

    call_order: list[str] = []
    read_order: list[str] = []

    async def fake_read(n: int) -> bytes:
        # Wait for both tasks to have queued before allowing either
        # read to proceed — gives the lock a chance to enforce
        # serialisation.
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
    # Unblock the reads.
    start_a.set()
    release_a.set()

    result_a, result_b = await asyncio.gather(fut_a, fut_b)

    # If serialised, A finishes before B starts its read.
    assert call_order.index("A_done") < call_order.index("B_done"), (
        f"DqliteProtocol must serialise RPCs; observed order {call_order!r}"
    )
    # Both calls succeeded with their expected payload — the lock
    # ensured each task's read pulled its OWN response from the
    # FIFO queue.
    assert result_a == (1, "10.0.0.1:9001")
    assert result_b == (2, "10.0.0.2:9001")
