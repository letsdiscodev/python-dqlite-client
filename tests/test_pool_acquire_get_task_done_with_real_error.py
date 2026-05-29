"""Pin ``acquire()``'s cancellation-cleanup branch where the inner
``_pool.get()`` task finished with an exception (not via cancellation) and
the outer ``acquire()`` is then cancelled: the helper must consume
``get_task.exception()`` so no "Task exception was never retrieved" fires.
"""

from __future__ import annotations

import asyncio
import gc
from unittest.mock import patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_acquire_consumes_get_task_exception_when_outer_cancelled() -> None:
    """``acquire()`` must consume the ``_pool.get()`` task's exception even
    while the outer task is cancelling, so no "Task exception was never
    retrieved" diagnostic fires.

    asyncio surfaces that diagnostic via ``loop.call_exception_handler``,
    not ``warnings.warn`` — a ``catch_warnings`` capture would miss it and
    pass against the regression, so capture via ``set_exception_handler``.
    """
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    # Slot exists but queue empty so acquire enters the wait-for-queue branch.
    pool._size = 1

    real_get = pool._pool.get

    async def boom_get() -> object:
        await asyncio.sleep(0)
        raise DqliteConnectionError("simulated transport failure")

    loop = asyncio.get_running_loop()
    captured: list[dict[str, object]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))
    try:
        with patch.object(pool._pool, "get", new=boom_get):
            acquire_task = loop.create_task(pool.acquire().__aenter__())
            # Let acquire enter the get-await, then cancel while it suspends.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            acquire_task.cancel()
            with pytest.raises((asyncio.CancelledError, DqliteConnectionError)):
                await acquire_task

        # Force GC and drain the loop so any orphaned-task diagnostic surfaces.
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_handler)

    leaked = [
        ctx
        for ctx in captured
        if "exception was never retrieved" in str(ctx.get("message", "")).lower()
    ]
    assert not leaked, f"Unretrieved exception diagnostics: {leaked}"

    with patch.object(pool._pool, "get", new=real_get):
        await pool.close()
