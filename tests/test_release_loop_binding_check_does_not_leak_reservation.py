"""Pin: ``ConnectionPool._release``'s loop-binding check runs INSIDE
the ``try`` arm so the ``finally`` releases the reservation even when
the check raises ``InterfaceError``.

Pre-fix the check ran BEFORE the try arm. A cross-loop / closed-loop
``_release`` call raised ``InterfaceError`` and bypassed the
compensating finally — the slot stayed permanently incremented.
Under operator-misuse-then-retry patterns (``engine.dispose()`` +
retry-on-new-loop) every misuse permanently shrank ``max_size`` by 1
until the pool exhausted.
"""

from __future__ import annotations

import asyncio
import weakref

import pytest

from dqliteclient.exceptions import InterfaceError
from dqliteclient.pool import ConnectionPool


def _make_pool_bound_to_dead_loop() -> ConnectionPool:
    """Build a pool whose loop binding references a now-closed loop.

    The bind happens at ``initialize`` / first ``acquire`` /
    first-loop touch; we forge it here by pointing ``_loop_ref`` at a
    fresh loop that we immediately close, then close the loop so the
    weakref becomes a "closed loop" diagnostic case.
    """
    pool = ConnectionPool(["localhost:9001"], max_size=2)
    dead_loop = asyncio.new_event_loop()
    pool._loop_ref = weakref.ref(dead_loop)
    dead_loop.close()
    return pool


@pytest.mark.asyncio
async def test_release_loop_mismatch_does_not_leak_reservation() -> None:
    """Pre-fix: ``_check_loop_binding`` raised before the try arm and
    the reservation stayed held. Post-fix: the check raises inside
    the try so the finally arm decrements ``_size``.
    """
    # Two distinct loops: the binding's loop is dead by the time the
    # call site (running on the live loop) enters _release. The
    # binding check raises InterfaceError.
    pool = _make_pool_bound_to_dead_loop()

    # Simulate a held reservation: hand a stub conn into _release.
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
