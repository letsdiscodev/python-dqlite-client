"""``open_admin_connection`` wraps dial+handshake under ``attempt_timeout`` so a
slow-handshaking peer cannot hang admin RPCs for the per-RPC ``timeout`` budget."""

import asyncio
import time

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_open_admin_connection_bounds_handshake_by_attempt_timeout() -> None:
    async def slow_dial(address: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        await asyncio.sleep(2.0)  # block past attempt_timeout
        raise AssertionError("should never reach")

    cluster = ClusterClient(
        MemoryNodeStore(["h:9001"]),
        timeout=10.0,
        dial_timeout=10.0,
        attempt_timeout=0.2,  # tight envelope
        dial_func=slow_dial,
    )

    started = time.monotonic()
    with pytest.raises(DqliteConnectionError):
        async with cluster.open_admin_connection("h:9001"):
            pytest.fail("should not reach")
    elapsed = time.monotonic() - started

    # Bounded by attempt_timeout, NOT the larger timeout / dial_timeout budget.
    assert elapsed < 1.0, (
        f"open_admin_connection took {elapsed:.2f}s; attempt_timeout was 0.2 s — envelope leaked"
    )
