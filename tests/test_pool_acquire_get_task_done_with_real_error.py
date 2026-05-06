"""Pin the pool's ``acquire()`` cancellation-cleanup branch where
the inner ``_pool.get()`` task completed with an exception (not via
cancellation), then the outer ``acquire()`` is cancelled.

The path at ``pool.py`` consumes ``get_task.exception()`` via a broad
``contextlib.suppress(BaseException)`` so the cancelled task isn't
reported as "Task exception was never retrieved" at GC. Today the
branch is uncovered (per ``--cov-report=term-missing``) — pin via a
parametric test.
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
    """``acquire()``'s except path must consume the ``_pool.get()``
    task's exception even when the outer task is cancelling, so no
    "Task exception was never retrieved" diagnostic is emitted by
    asyncio's loop exception handler.

    asyncio surfaces "Task exception was never retrieved" via
    ``loop.call_exception_handler`` (i.e. through the loop's
    exception handler / asyncio logger), NOT via ``warnings.warn``.
    Use ``set_exception_handler`` to capture the diagnostic — a
    ``warnings.catch_warnings`` capture would silently miss it and
    the pin would pass against the regression it is meant to prevent.
    Same pattern as ``test_close_impl_reaps_pending_drain_created_during_await.py``.
    """
    pool = ConnectionPool(["localhost:9001"], min_size=0, max_size=1)
    # Force the pool to think a slot exists and the queue is empty
    # so ``acquire()`` enters the wait-for-queue branch.
    pool._size = 1

    # Fake the pool's wait-for-conn task by raising OperationalError
    # synchronously when the await happens. Patch the queue.get so we
    # can drive both states.
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
            # Yield to let acquire enter the get-await.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Cancel the outer acquire while the inner get is suspending.
            acquire_task.cancel()
            with pytest.raises((asyncio.CancelledError, DqliteConnectionError)):
                await acquire_task

        # Force GC so any orphaned-task diagnostic surfaces.
        gc.collect()
        # Let the loop drain so any deferred handler call lands.
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_handler)

    leaked = [
        ctx
        for ctx in captured
        if "exception was never retrieved" in str(ctx.get("message", "")).lower()
    ]
    assert not leaked, f"Unretrieved exception diagnostics: {leaked}"

    # Cleanup
    with patch.object(pool._pool, "get", new=real_get):
        await pool.close()
