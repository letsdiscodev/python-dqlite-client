"""When ``acquire()`` dequeues a dead conn and must create a
REPLACEMENT, the replacement create is clamped to the remaining
acquire deadline. Symmetric with the fresh-slot clamp test at
``tests/test_pool_acquire_reservation_lifecycle.py::test_acquire_create_connection_clamped_by_pool_timeout``
— exercises the OTHER ``asyncio.timeout`` arm in ``pool.py``
(the dead-conn-replacement at ``pool.py:1502-1532``).

Regressions silent without a pin:

- Inverted deadline check (``> 0`` instead of ``<= 0``) would skip the
  raise and proceed to ``asyncio.timeout(negative)`` which is
  loop-dependent.
- A refactor moving the shielded ``_release_reservation`` outside the
  ``except`` arm would leak a reservation slot on every timed-out
  replacement, eventually saturating the pool.
- Drift in the ``DqliteConnectionError`` message shape would lose the
  forensic detail operators read while investigating pool saturation
  (max_size / idle / checked_out / timeout).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_replacement_create_clamped_by_pool_timeout() -> None:
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)
    # Seed the pool with a dead conn so acquire takes the replacement
    # arm. The seeded conn must look "dead" for the
    # ``not conn.is_connected`` branch to fire.
    dead = MagicMock()
    dead.is_connected = False
    dead.close = AsyncMock()
    dead._pool_released = True
    pool._pool.put_nowait(dead)
    # The seeded conn occupies one reservation slot.
    pool._size = 1

    async def _slow_create() -> Any:
        # Sleeps far longer than pool.timeout so the clamp is
        # observable.
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied to replacement create")

    loop = asyncio.get_running_loop()
    started = loop.time()

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        patch.object(pool, "_drain_idle", new=AsyncMock()),
        pytest.raises(DqliteConnectionError, match="Timed out creating"),
    ):
        async with pool.acquire():
            pytest.fail("should not reach the with-body")

    elapsed = loop.time() - started
    assert elapsed < 0.6, (
        f"acquire() blocked for {elapsed:.3f}s under pool.timeout=0.1s on "
        f"the replacement create path; the asyncio.timeout clamp at "
        f"pool.py:1515 did not fire"
    )


@pytest.mark.asyncio
async def test_acquire_replacement_create_timeout_message_carries_pool_state() -> None:
    """The DqliteConnectionError raised by the replacement-create
    timeout arm carries pool-id, max_size, idle, checked_out and
    timeout for operator forensics. A regression dropping any of those
    fields would erase the only signal an SRE has for pool saturation
    diagnosis."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.1)
    dead = MagicMock()
    dead.is_connected = False
    dead.close = AsyncMock()
    dead._pool_released = True
    pool._pool.put_nowait(dead)
    pool._size = 1

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied")

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        patch.object(pool, "_drain_idle", new=AsyncMock()),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with pool.acquire():
            pytest.fail("should not reach")

    msg = str(exc_info.value)
    assert "Timed out creating a fresh connection from the pool" in msg
    assert "max_size=1" in msg
    assert "idle=" in msg
    assert "checked_out=" in msg
    assert "timeout=0.1s" in msg
