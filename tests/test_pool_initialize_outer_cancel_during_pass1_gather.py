"""Pin: ``initialize``'s finally pass-2 (close completed tasks' connections)
still runs even when pass-1's cleanup gather raises CancelledError."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_outer_cancel_during_pass1_gather_still_closes_completed_conns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    completed_conns: list[_FakeConn] = []

    block_event = asyncio.Event()
    create_call = {"n": 0}

    async def fake_create_connection() -> Any:
        n = create_call["n"]
        create_call["n"] += 1
        if n < 2:
            c = _FakeConn()
            completed_conns.append(c)
            return c
        await block_event.wait()
        raise AssertionError("blocking task resumed unexpectedly")

    pool._create_connection = fake_create_connection

    # Make both the success-path gather and the finally's pass-1 cleanup gather
    # raise CancelledError, mimicking an outer asyncio.timeout firing.
    real_gather = asyncio.gather
    gather_calls = {"n": 0}

    async def patched_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["n"] += 1
        if gather_calls["n"] == 1:
            # Success-path gather: let the first two creates land, then raise
            # CancelledError to enter the finally with one task still pending.
            for t in args:
                if isinstance(t, asyncio.Task) and not t.done():
                    pass
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise asyncio.CancelledError("synthetic-outer-cancel-during-gather")
        if gather_calls["n"] == 2:
            # Pass-1 cleanup gather: await each pending task, then raise.
            for t in args:
                if isinstance(t, asyncio.Task) and not t.done():
                    t.cancel()
                    with contextlib.suppress(BaseException):
                        await t
            raise asyncio.CancelledError("synthetic-outer-cancel-during-pass1")
        return await real_gather(*args, **kwargs)

    monkeypatch.setattr("asyncio.gather", patched_gather)

    with pytest.raises(asyncio.CancelledError):
        await pool.initialize()

    assert len(completed_conns) == 2, (
        f"test wired wrong: expected 2 completed conns, got {len(completed_conns)}"
    )
    for c in completed_conns:
        assert c.closed, (
            "completed-task connection was orphaned when outer cancel hit "
            "pass-1's gather (pass-2 close walk was skipped)"
        )

    pool._create_connection = AsyncMock(side_effect=AssertionError)
