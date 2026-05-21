"""Pin: ``ConnectionPool.initialize``'s finally-arm ``raise
outer_cancel`` chains the try-body in-flight exception explicitly on
``__cause__``.

The finally arm runs a cleanup gather over still-pending create tasks;
if an outer ``BaseException`` propagates into that cleanup gather, the
captured exception is stored on ``outer_cancel`` and re-raised. Pre-fix
the re-raise had no ``from`` clause, so ``__cause__`` was ``None`` and
the try-body exception that initiated the unwind sat only on
``__context__``. SA's ``walk_cause_chain`` follows ``__cause__`` (plus
PEP 654 group children) and never descends into ``__context__`` — the
leaf cause needed for is_disconnect classification was invisible.

This test drives the load-bearing path: the success-path gather raises
a synthetic ``DqliteConnectionError`` (the try-body root cause), then
the finally's cleanup gather raises ``CancelledError`` (the outer
cancel mid-cleanup). The finally's ``raise outer_cancel`` lands at the
user with the ``DqliteConnectionError`` reachable via ``__cause__``.
"""

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

    # Two create_tasks: the first returns fast, the second blocks
    # forever — so when the success-path gather raises, the second
    # task is still pending and lands in ``cancelled_pending``.
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

    # The try-body in-flight exception we want to see on ``__cause__``
    # of the user-visible re-raise.
    try_body_exc = DqliteConnectionError("synthetic-try-body-transport-broken")

    gather_calls = {"n": 0}

    async def patched_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["n"] += 1
        if gather_calls["n"] == 1:
            # Success-path gather: yield twice so the first task
            # lands its result, then raise ``DqliteConnectionError``
            # out of the gather frame. This is the try-body
            # in-flight exception that drives entry into the
            # finally with ``gather_returned == False``.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise try_body_exc
        # Finally-arm cleanup gather: mimic outer cancel firing
        # during the cancelled-pending await.
        for t in args:
            if isinstance(t, asyncio.Task) and not t.done():
                t.cancel()
                with contextlib.suppress(BaseException):
                    await t
        raise asyncio.CancelledError("synthetic-outer-cancel-during-cleanup-gather")

    monkeypatch.setattr("asyncio.gather", patched_gather)

    with pytest.raises(BaseException) as excinfo:
        await pool.initialize()

    # Walk the ``__cause__`` chain (SA's discipline). The try-body
    # ``DqliteConnectionError`` must be reachable via ``__cause__``,
    # NOT only via ``__context__``.
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
    # The user-visible exception is the outer cancel (CancelledError);
    # its ``__cause__`` should be the captured try-body exception.
    assert isinstance(excinfo.value, asyncio.CancelledError)
    assert isinstance(excinfo.value.__cause__, DqliteConnectionError)

    # Cleanup for the pool fixture.
    pool._create_connection = AsyncMock(side_effect=AssertionError)


@pytest.mark.asyncio
async def test_finally_outer_cancel_self_cycle_guard_falls_back_to_plain_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling: when the in-flight is the same object as
    ``outer_cancel`` (self-cycle), the finally falls back to plain
    ``raise outer_cancel`` rather than ``raise X from X``. Drive by
    raising the SAME exception object from both the success-path
    gather (becomes in_flight) and the cleanup gather (becomes
    outer_cancel)."""
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

    # The self-cycle guard fires: ``in_flight is outer_cancel`` so the
    # ``raise outer_cancel from in_flight`` arm is skipped and the
    # plain ``raise outer_cancel`` runs. ``__cause__`` is unchanged
    # (None, since the original raise had no ``from``).
    assert excinfo.value is shared_exc
    assert excinfo.value.__cause__ is None

    pool._create_connection = AsyncMock(side_effect=AssertionError)
