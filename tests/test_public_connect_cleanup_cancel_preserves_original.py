"""Pin: ``dqliteclient.connect()`` cleanup-close shields against a
fresh outer ``CancelledError`` during ``conn.close()`` so the original
connect-time failure stays the active exception (``except Exception``
does not catch ``CancelledError``, which would otherwise supplant it).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import dqliteclient
from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_cancel_during_cleanup_close_does_not_supplant_original() -> None:
    """A cancel during cleanup-close keeps the original OSError active."""

    close_started = asyncio.Event()
    close_finished = asyncio.Event()

    async def _slow_close(self: DqliteConnection) -> None:
        close_started.set()
        # The test cancels while we are suspended here; the shield must
        # keep us alive to completion.
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            close_finished.set()
            raise
        close_finished.set()

    async def _drive() -> None:
        with (
            patch.object(
                DqliteConnection,
                "connect",
                new=AsyncMock(side_effect=OSError("connect-failed")),
            ),
            patch.object(DqliteConnection, "close", new=_slow_close),
        ):
            await dqliteclient.connect("localhost:9001", database="test", timeout=5.0)

    task = asyncio.create_task(_drive())
    await close_started.wait()
    task.cancel()

    with pytest.raises(BaseException) as exc_info:
        await task

    raised = exc_info.value
    if isinstance(raised, OSError):
        assert "connect-failed" in str(raised)
    else:
        pytest.fail(
            f"expected OSError('connect-failed') to be the active exception "
            f"after cancel-during-cleanup, got {type(raised).__name__}: {raised!r} "
            f"(context={raised.__context__!r})"
        )
