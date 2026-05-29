"""Pin that ``acquire()``'s cleanup arm absorbs ``_POOL_CLEANUP_EXCEPTIONS`` (OSError/
TimeoutError) from the shielded ``conn.close()`` on the closed-pool branch and DEBUG-logs,
so the user's original body exception propagates cleanly instead of being masked by teardown."""

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
                # close() raises OSError so the cleanup arm's suppression must fire.
                conn.close = AsyncMock(side_effect=OSError("transport gone"))
                # Simulate concurrent pool.close(): routes cleanup through conn.close().
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
