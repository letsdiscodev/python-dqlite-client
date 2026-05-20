"""Pin: ``ConnectionPool.acquire``'s dead-conn pre-drain close
absorbs ``RuntimeError`` and releases the reservation slot on any
``BaseException`` escape so ``_size`` stays consistent.

The narrow ``contextlib.suppress(*_POOL_CLEANUP_EXCEPTIONS)``
around ``conn.close()`` at the dead-conn arm catches the documented
transport-class failures (OSError, DqliteConnectionError,
ProtocolError, OperationalError, InterfaceError) but did NOT catch
``RuntimeError``. ``DqliteConnection.close()`` raises
``RuntimeError`` on two well-documented paths:

* "Event loop is closed" during racing ``engine.dispose()``.
* Cross-loop ``asyncio.Lock`` from a stale conn reference held
  across an ``asyncio.run`` boundary.

The sibling broken-conn arm further down in ``acquire()``
explicitly catches ``RuntimeError`` with the same rationale. The
dead-conn arm was the asymmetric sibling — and a ``RuntimeError``
escape there leaks one reservation slot per occurrence, since the
``_drain_idle`` / ``_create_connection`` cleanup arms below would
never run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.pool import ConnectionPool


def _make_dead_looking_conn() -> MagicMock:
    """A conn that the dead-conn arm will detect:
    ``not conn.is_connected`` short-circuits before
    ``_socket_looks_dead`` so this is the simpler arm to trigger."""
    conn = MagicMock()
    conn.is_connected = False
    conn._pool_released = True  # set by ``_release`` on enqueue
    conn._protocol = None
    return conn


@pytest.mark.asyncio
async def test_dead_conn_close_runtime_error_releases_reservation() -> None:
    """A ``RuntimeError`` from the dead-conn arm's ``close()`` must
    not escape ``acquire()`` AND must release the reservation slot
    before the create-connection attempt runs (which itself will
    fail, but ``_size`` after the failure must be unchanged from
    before the acquire — no leak)."""
    pool = ConnectionPool(["127.0.0.1:9001"], min_size=0, max_size=2, timeout=1.0)

    queued = _make_dead_looking_conn()

    async def _boom_close() -> None:
        raise RuntimeError("Event loop is closed")

    queued.close = AsyncMock(side_effect=_boom_close)
    pool._pool.put_nowait(queued)
    pool._size = 1  # reservation slot for the queued conn

    # Make _create_connection a no-op-raise so the test does not try
    # to dial; the assertion is about reservation accounting on the
    # RuntimeError path, not about the eventual create outcome.
    async def _fail_create() -> object:
        raise RuntimeError("test: create_connection short-circuited")

    pool._create_connection = AsyncMock(side_effect=_fail_create)
    # _drain_idle on an empty queue is a no-op.

    size_before = pool._size
    with pytest.raises(RuntimeError):
        async with pool.acquire():
            pytest.fail("acquire should not yield a connection")
    size_after = pool._size

    # PRE-FIX: dead-conn close RuntimeError propagated past the
    # finally, skipped _drain_idle / _create_connection, leaving
    # the reservation slot inflated by 1.
    # POST-FIX: the dead-conn close arm absorbs RuntimeError and
    # proceeds; _create_connection's failure arm releases the
    # reservation; size goes back to baseline.
    assert size_after <= size_before - 1, (
        f"dead-conn close RuntimeError must release the reservation: "
        f"size_before={size_before}, size_after={size_after}"
    )
