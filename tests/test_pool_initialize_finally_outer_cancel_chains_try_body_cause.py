"""Pin: ``initialize``'s finally-arm ``raise outer_cancel`` chains the try-body
exception on ``__cause__`` (SA's ``walk_cause_chain`` ignores ``__context__``,
so the leaf cause for is_disconnect classification must be on ``__cause__``)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_finally_outer_cancel_chains_try_body_cause_on_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ConnectionPool(["localhost:19001"], min_size=2, max_size=2, timeout=0.5)

    # First task returns fast, second blocks forever so it lands pending
    # in ``cancelled_pending`` when the success-path gather raises.
    block_event = asyncio.Event()
    create_call = {"n": 0}

    async def fake_create_connection() -> Any:
        n = create_call["n"]
        create_call["n"] += 1
        if n == 0:
            return _FakeConn()
        await block_event.wait()
        raise AssertionError("blocking task resumed unexpectedly")

    pool._create_connection = fake_create_connection

    # The try-body in-flight exception we want to see on ``__cause__``.
    try_body_exc = DqliteConnectionError("synthetic-try-body-transport-broken")

    gather_calls = {"n": 0}

    async def patched_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["n"] += 1
        if gather_calls["n"] == 1:
            # Success-path gather: let the first task land, then raise the
            # try-body exception so the finally runs with gather_returned False.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise try_body_exc
        # Finally-arm cleanup gather: mimic outer cancel during the await.
        for t in args:
            if isinstance(t, asyncio.Task) and not t.done():
                t.cancel()
                with contextlib.suppress(BaseException):
                    await t
        raise asyncio.CancelledError("synthetic-outer-cancel-during-cleanup-gather")

    monkeypatch.setattr("asyncio.gather", patched_gather)

    with pytest.raises(BaseException) as excinfo:
        await pool.initialize()

    # The try-body DqliteConnectionError must be reachable via __cause__.
    chain: list[type[BaseException]] = []
    cur: BaseException | None = excinfo.value
    visited: set[int] = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        chain.append(type(cur))
        cur = cur.__cause__
    assert any(issubclass(t, DqliteConnectionError) for t in chain), (
        f"try-body DqliteConnectionError is not reachable via __cause__ "
        f"chain; chain={[t.__name__ for t in chain]}. The finally arm "
        f"raised outer_cancel without chaining the in-flight exception."
    )
    assert isinstance(excinfo.value, asyncio.CancelledError)
    assert isinstance(excinfo.value.__cause__, DqliteConnectionError)

    pool._create_connection = AsyncMock(side_effect=AssertionError)


@pytest.mark.asyncio
async def test_finally_outer_cancel_self_cycle_guard_falls_back_to_plain_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-cycle: when in-flight is the same object as outer_cancel, the
    finally falls back to plain ``raise`` rather than ``raise X from X``."""
    pool = ConnectionPool(["localhost:19001"], min_size=2, max_size=2, timeout=0.5)

    block_event = asyncio.Event()
    create_call = {"n": 0}

    async def fake_create_connection() -> Any:
        n = create_call["n"]
        create_call["n"] += 1
        if n == 0:
            return _FakeConn()
        await block_event.wait()
        raise AssertionError("blocking task resumed unexpectedly")

    pool._create_connection = fake_create_connection

    shared_exc = asyncio.CancelledError("shared-outer-cancel")
    gather_calls = {"n": 0}

    async def patched_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["n"] += 1
        if gather_calls["n"] == 1:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise shared_exc
        for t in args:
            if isinstance(t, asyncio.Task) and not t.done():
                t.cancel()
                with contextlib.suppress(BaseException):
                    await t
        raise shared_exc

    monkeypatch.setattr("asyncio.gather", patched_gather)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await pool.initialize()

    # Self-cycle guard skips ``raise X from X``; __cause__ stays None.
    assert excinfo.value is shared_exc
    assert excinfo.value.__cause__ is None

    pool._create_connection = AsyncMock(side_effect=AssertionError)
