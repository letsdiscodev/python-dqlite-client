"""Pin: pool ``acquire``'s broken-conn cleanup shields ``conn.close()``.

The exception-cleanup branch in ``acquire`` runs when the user's
checked-out connection is broken (invalidated by execute/fetch error).
It calls ``await conn.close()`` to release the transport. If an outer
``asyncio.timeout`` or ``CancelScope`` fires during the cleanup, the
unshielded close is interrupted and the user's transport leaks open
until GC. The fix wraps the close in ``asyncio.shield`` and
``contextlib.suppress(CancelledError)`` so the close completes
(bounded by ``_close_timeout``) before the cancellation propagates;
the inner ``finally`` then guarantees ``_pool_released = True``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


async def _build_pool_with_breakable_conn() -> tuple[ConnectionPool, DqliteConnection]:
    """Construct a pool whose ``_create_connection`` returns a conn
    that LOOKS connected at acquire time but transitions to broken
    once the user invalidates it (simulating leader flip mid-execute).
    """
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = DqliteConnection("localhost:9001")
    # Pretend the conn is fully connected so ``acquire`` does not run
    # the entry-time ``_drain_idle`` (which would mask the cleanup
    # branch we are testing).
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    assert conn.is_connected is True

    async def _fake_create_connection() -> DqliteConnection:
        return conn

    pool._create_connection = _fake_create_connection
    return pool, conn


def _break_conn(conn: DqliteConnection) -> None:
    """Simulate a leader-flip-style invalidation: drop ``_protocol``
    so ``conn.is_connected`` flips to False, sending the cleanup down
    the broken-conn branch."""
    conn._protocol = None
    conn._db_id = None


@pytest.mark.asyncio
async def test_acquire_cleanup_close_completes_under_outer_cancel() -> None:
    """Cancel during the broken-conn cleanup must NOT skip
    ``conn.close()``: shield + suppress(CancelledError) keeps the
    close running while the original cancel propagates via the
    surrounding ``raise``. Pin via ``_pool_released=True`` (set in the
    inner ``finally`` after the close) — without the fix it would
    stay False because cancel propagated past the bare close."""
    pool, conn = await _build_pool_with_breakable_conn()

    close_completed = False

    async def fake_close() -> None:
        nonlocal close_completed
        # Yield once so a concurrent cancel can land mid-close. The
        # shield must keep us running through the second statement.
        await asyncio.sleep(0)
        # Without the shield, the cancel queued via raise CancelledError
        # below would propagate through this await and skip the
        # assignment — the test would catch that as close_completed=False.
        close_completed = True

    async def fake_drain_idle() -> None:
        return

    pool._drain_idle = fake_drain_idle

    with (
        patch.object(conn, "close", new=fake_close),
        pytest.raises((asyncio.CancelledError, ValueError)),
    ):
        async with pool.acquire():
            _break_conn(conn)
            # Schedule a cancel of the current task so it lands
            # while ``conn.close`` is suspended on the
            # ``await asyncio.sleep(0)`` above. The shield must
            # absorb the cancel and let close finish.
            current = asyncio.current_task()
            assert current is not None
            asyncio.get_running_loop().call_soon(current.cancel)
            raise ValueError("user code error")

    assert close_completed, (
        "conn.close() must run to completion despite cancel — shield "
        "around the close await keeps it running"
    )
    assert conn._pool_released is True, (
        "_pool_released must be set in inner finally — the shield+suppress "
        "around close ensures we reach the flag-set"
    )


@pytest.mark.asyncio
async def test_acquire_cleanup_close_failure_still_sets_pool_released() -> None:
    """If ``conn.close()`` raises an unhandled exception (not
    OSError/DqliteConnectionError), the inner ``finally`` must still
    set ``_pool_released=True`` so subsequent close() short-circuits.

    Also pin: the user's ORIGINAL ValueError must propagate — the
    close-time RuntimeError must NOT supplant it. The widened
    ``except Exception`` log-and-absorb pattern preserves the user's
    exception for the bare ``raise`` further out."""
    pool, conn = await _build_pool_with_breakable_conn()

    async def fake_close() -> None:
        # Simulate an unrecognised close failure (not OSError /
        # DqliteConnectionError). Must NOT supplant the user's
        # original exception.
        raise RuntimeError("simulated close-time RuntimeError")

    async def fake_drain_idle() -> None:
        return

    pool._drain_idle = fake_drain_idle

    with patch.object(conn, "close", new=fake_close), pytest.raises(ValueError, match="user"):
        async with pool.acquire():
            _break_conn(conn)
            raise ValueError("user code error")

    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_drain_idle_failure_does_not_skip_close() -> None:
    """Pin existing behaviour: the bare ``_drain_idle()`` failure path
    is intentionally narrow (DqliteConnectionError / OSError swallowed
    via _POOL_CLEANUP_EXCEPTIONS); the close still runs after."""
    pool, conn = await _build_pool_with_breakable_conn()

    close_call_count = 0

    async def fake_close() -> None:
        nonlocal close_call_count
        close_call_count += 1

    async def fake_drain_idle() -> None:
        raise DqliteConnectionError("simulated drain failure")

    pool._drain_idle = fake_drain_idle

    with patch.object(conn, "close", new=fake_close), pytest.raises(ValueError):
        async with pool.acquire():
            _break_conn(conn)
            raise ValueError("user code error")

    assert close_call_count == 1
    assert conn._pool_released is True
