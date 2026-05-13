"""``ConnectionPool.initialize``'s unqueued-survivor cleanup walks
connections that completed creation but were never queued (e.g. due to
a ``put_nowait`` failure mid-init). The loop has two except arms; this
file pins the **CancelledError** arm at ``pool.py:884-892``.

The sibling ``_POOL_CLEANUP_EXCEPTIONS`` arm is covered by
``test_pool_initialize_unqueued_survivor_close_log.py``. The cancel arm
encodes the "continue closing the remaining survivors before the
cancel resumes" guarantee — without a pin, a future refactor that
collapses both arms into one and raises on first cancel would
silently leak the remaining conns and erase the operator-facing
DEBUG breadcrumb.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_unqueued_survivor_cancel_continues_closing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CancelledError landing inside the shielded close of one
    survivor must (1) emit the documented DEBUG breadcrumb and (2)
    continue closing the remaining survivors before the cancel resumes
    via the loop's natural exit."""
    pool = ConnectionPool(addresses=["h:9001"], min_size=2, max_size=4)

    # Two conns: one whose close() raises CancelledError mid-call, one
    # that closes cleanly afterwards. The second close MUST be awaited
    # for the "continue closing the remaining survivors" guarantee.
    cancelled_conn = MagicMock()
    cancelled_conn.close = AsyncMock(side_effect=asyncio.CancelledError())
    clean_conn = MagicMock()
    clean_conn.close = AsyncMock()

    create_calls = iter([cancelled_conn, clean_conn])

    async def stub_create() -> MagicMock:
        return next(create_calls)

    with (
        patch.object(pool, "_create_connection", new=stub_create),
        patch.object(pool._pool, "put_nowait", side_effect=asyncio.QueueFull),
        caplog.at_level("DEBUG"),
        pytest.raises(asyncio.QueueFull),
    ):
        await pool.initialize()

    # Both close() calls must have been attempted. The CancelledError on
    # the first must NOT have aborted the loop.
    cancelled_conn.close.assert_awaited()
    clean_conn.close.assert_awaited()

    # The DEBUG breadcrumb on the cancel arm is the operator-facing
    # forensic trail. A regression that drops the log line would erase
    # the only signal a partial-init abort produces.
    assert any("cancel during unqueued-survivor close" in r.getMessage() for r in caplog.records), (
        "expected the cancel-arm DEBUG breadcrumb at pool.py:889 "
        "'cancel during unqueued-survivor close'"
    )
