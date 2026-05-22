"""Pin: ``acquire()``'s ``except BaseException:`` cleanup arm
suppresses ``_POOL_CLEANUP_EXCEPTIONS`` from the shielded
``conn.close()`` on the closed-pool branch and DEBUG-logs.

The cited code path at pool.py:1971-1979 (and the sibling QueueFull
+ reset-fail branches at 1983-1991, 1995-2003) is reached when a
user-code raise from inside ``async with pool.acquire() as conn:``
coincides with ``pool.close()`` completing concurrently. The
inline ``_POOL_CLEANUP_EXCEPTIONS`` suppression catches OSError /
TimeoutError from the close and absorbs them so the user's
original exception can propagate cleanly. Without coverage, a
regression that narrowed the catch (or removed the suppression)
would let pool-tear-down noise mask the body raise.

Integration-style: build a real ``ConnectionPool`` against the
live cluster, take a connection, mock its ``.close`` to raise
``OSError``, flip ``pool._closed`` to True (simulate concurrent
``close()``), then raise from the body.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from dqliteclient import create_pool

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_acquire_cleanup_arm_suppresses_oserror_from_close_on_closed_pool(
    cluster_address: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = await create_pool([cluster_address], min_size=1, max_size=1, timeout=5.0)
    try:
        with (
            caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"),
            pytest.raises(RuntimeError, match="user-body raise"),
        ):
            async with pool.acquire() as conn:
                # Mock conn.close to raise OSError so the cleanup
                # arm's ``_POOL_CLEANUP_EXCEPTIONS`` suppression
                # fires when the closed-pool branch awaits it.
                conn.close = AsyncMock(side_effect=OSError("transport gone"))
                # Simulate ``pool.close()`` completing
                # concurrently: flip ``_closed`` so the cleanup
                # arm's healthy-branch + ``_closed`` re-check
                # routes through ``conn.close()``.
                pool._closed = True
                raise RuntimeError("user-body raise")
        debug_lines = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelname == "DEBUG"
            and "pool.acquire cleanup: conn.close(" in rec.getMessage()
            and "failed" in rec.getMessage()
        ]
        assert debug_lines, (
            "Broken-connection cleanup arm must DEBUG-log the absorbed "
            "OSError from conn.close(); got log records "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
        )
    finally:
        # Drop the closed flag so pool.close() runs the inner drain.
        pool._closed = False
        await pool.close()
