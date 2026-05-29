"""Pin: pool ``acquire``'s broken-conn cleanup shields ``conn.close()``.

An outer cancel landing during cleanup must not interrupt the close and
leak the user's transport; shield + suppress(CancelledError) keep the
close running, then the inner finally sets ``_pool_released = True``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


async def _build_pool_with_breakable_conn() -> tuple[ConnectionPool, DqliteConnection]:
    """Pool whose conn looks connected at acquire but later breaks."""
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    conn = DqliteConnection("localhost:9001")
    # Look fully connected so acquire skips entry-time _drain_idle, which
    # would mask the cleanup branch under test.
    conn._protocol = object()  # type: ignore[assignment]
    conn._db_id = 1
    assert conn.is_connected is True

    async def _fake_create_connection() -> DqliteConnection:
        return conn

    pool._create_connection = _fake_create_connection
    return pool, conn


def _break_conn(conn: DqliteConnection) -> None:
    """Drop ``_protocol`` so ``is_connected`` flips to the broken branch."""
    conn._protocol = None
    conn._db_id = None


@pytest.mark.asyncio
async def test_acquire_cleanup_close_completes_under_outer_cancel() -> None:
    """Cancel during cleanup must not skip ``conn.close()``; pinned via
    ``_pool_released=True``, set in the inner finally after the close."""
    pool, conn = await _build_pool_with_breakable_conn()

    close_completed = False

    async def fake_close() -> None:
        nonlocal close_completed
        # Yield so a concurrent cancel lands mid-close; the shield must
        # keep us running through the assignment below.
        await asyncio.sleep(0)
        close_completed = True

    async def fake_drain_idle(*_args: object, **_kwargs: object) -> None:
        return

    pool._drain_idle = fake_drain_idle

    with (
        patch.object(conn, "close", new=fake_close),
        pytest.raises((asyncio.CancelledError, ValueError)),
    ):
        async with pool.acquire():
            _break_conn(conn)
            # Cancel this task so it lands while conn.close is suspended
            # on its sleep(0); the shield must absorb it and let close finish.
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
    """An unhandled close() exception must still set ``_pool_released``,
    and the user's original ValueError must propagate (not the close-time
    RuntimeError)."""
    pool, conn = await _build_pool_with_breakable_conn()

    async def fake_close() -> None:
        raise RuntimeError("simulated close-time RuntimeError")

    async def fake_drain_idle(*_args: object, **_kwargs: object) -> None:
        return

    pool._drain_idle = fake_drain_idle

    with patch.object(conn, "close", new=fake_close), pytest.raises(ValueError, match="user"):
        async with pool.acquire():
            _break_conn(conn)
            raise ValueError("user code error")

    assert conn._pool_released is True


@pytest.mark.asyncio
async def test_drain_idle_failure_does_not_skip_close() -> None:
    """A swallowed ``_drain_idle()`` failure must not skip the close."""
    pool, conn = await _build_pool_with_breakable_conn()

    close_call_count = 0

    async def fake_close() -> None:
        nonlocal close_call_count
        close_call_count += 1

    async def fake_drain_idle(*_args: object, **_kwargs: object) -> None:
        raise DqliteConnectionError("simulated drain failure")

    pool._drain_idle = fake_drain_idle

    with patch.object(conn, "close", new=fake_close), pytest.raises(ValueError):
        async with pool.acquire():
            _break_conn(conn)
            raise ValueError("user code error")

    assert close_call_count == 1
    assert conn._pool_released is True
