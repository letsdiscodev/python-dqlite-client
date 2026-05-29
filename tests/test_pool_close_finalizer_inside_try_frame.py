"""close() must keep _finalizer.detach() inside the try frame: a BaseException
between _closed=True and the first awaited line would otherwise skip the finally
arm (_close_done.set()), hanging a second caller that observes _closed=True."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_finalizer_detach_in_close_lives_inside_try_frame() -> None:
    """A BaseException from finalizer.detach() must still leave _close_done set."""

    pool = ConnectionPool(addresses=["10.0.0.1:9001"], min_size=0, max_size=1)
    pool._closed_event = asyncio.Event()
    sentinel = SystemExit("synthetic")

    class _BoomFinalizer:
        def detach(self) -> None:
            raise sentinel

    pool._finalizer = _BoomFinalizer()  # type: ignore[assignment]

    with pytest.raises(SystemExit) as excinfo:
        await pool.close()

    assert excinfo.value is sentinel
    assert pool._closed is True
    assert pool._close_done is not None
    assert pool._close_done.is_set(), (
        "close()'s try/finally must set _close_done even when the "
        "body raises BaseException, so the second-caller early-return "
        "arm at the top of close() does not hang on _close_done.wait()."
    )


@pytest.mark.asyncio
async def test_second_caller_does_not_hang_when_first_caller_raised() -> None:
    """First close() caller raises mid-body; the second caller must still return."""

    pool = ConnectionPool(addresses=["10.0.0.1:9001"], min_size=0, max_size=1)
    pool._closed_event = asyncio.Event()

    class _BoomFinalizer:
        def detach(self) -> None:
            raise SystemExit("synthetic")

    pool._finalizer = _BoomFinalizer()  # type: ignore[assignment]

    with pytest.raises(SystemExit):
        await pool.close()

    # Tight timeout so a regression fails fast instead of hanging the session.
    await asyncio.wait_for(pool.close(), timeout=2.0)
