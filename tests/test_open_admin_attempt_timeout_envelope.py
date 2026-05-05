"""Pin: ``ClusterClient.open_admin_connection`` wraps the entire
dial+handshake under ``attempt_timeout`` so a slow-handshaking
peer cannot hang admin RPCs for the per-RPC ``timeout`` budget.

Mirror of ``_connect_impl`` (SQL path) and Go canonical
``NewDirectConnector.Connect`` (``connector.go:148-150``).
"""

import asyncio
import time

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_open_admin_connection_bounds_handshake_by_attempt_timeout() -> None:
    """A peer that handshakes slowly must not hold the admin
    surface for the per-RPC ``timeout`` budget — the
    ``attempt_timeout`` envelope around the dial+handshake bounds
    the wall clock."""

    async def slow_dial(address: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        # Block past attempt_timeout.
        await asyncio.sleep(2.0)
        raise AssertionError("should never reach")

    cluster = ClusterClient(
        MemoryNodeStore(["h:9001"]),
        timeout=10.0,  # generous per-RPC budget
        dial_timeout=10.0,  # generous dial budget
        attempt_timeout=0.2,  # tight envelope
        dial_func=slow_dial,
    )

    started = time.monotonic()
    with pytest.raises(DqliteConnectionError):
        async with cluster.open_admin_connection("h:9001"):
            pytest.fail("should not reach")
    elapsed = time.monotonic() - started

    # The wall-clock must be bounded by attempt_timeout (with slack)
    # — NOT by the larger ``timeout`` / ``dial_timeout`` budget.
    assert elapsed < 1.0, (
        f"open_admin_connection took {elapsed:.2f}s; attempt_timeout was 0.2 s — envelope leaked"
    )
