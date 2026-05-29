"""Pin: ``_release``'s loop-binding check runs inside the ``try`` so
the ``finally`` releases the reservation even when it raises
``InterfaceError`` (else repeated misuse leaks slots and shrinks the pool)."""

from __future__ import annotations

import asyncio
import weakref

import pytest

from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import ConnectionPool


def _make_pool_bound_to_dead_loop() -> ConnectionPool:
    """Build a pool whose loop binding references a now-closed loop."""
    pool = ConnectionPool(["localhost:9001"], max_size=2)
    dead_loop = asyncio.new_event_loop()
    pool._loop_ref = weakref.ref(dead_loop)
    dead_loop.close()
    return pool


@pytest.mark.asyncio
async def test_release_loop_mismatch_does_not_leak_reservation() -> None:
    pool = _make_pool_bound_to_dead_loop()

    class _StubConn:
        _pool_released = False
        _address = "stub:9001"
        _pending_drain = None

        async def close(self) -> None:
            return None

    pool._size = 1
    starting = pool._size

    with pytest.raises(InterfaceError):
        await pool._release(_StubConn())  # type: ignore[arg-type]

    assert pool._size == starting - 1, (
        f"_release's loop-binding check must run inside the try arm "
        f"so the finally decrements the slot even on InterfaceError; "
        f"_size={pool._size} (expected {starting - 1})"
    )
