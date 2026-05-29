"""``_run_protocol`` invalidates the connection when a ``BaseExceptionGroup``
carries a cancel-class child: ``isinstance(eg, CancelledError)`` is False (PEP
654), so without an explicit group arm the wire state would be silently reused."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


def _make_connection() -> DqliteConnection:
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._protocol = MagicMock()
    conn._db_id = 1
    conn._in_use = False
    conn._closed = False
    conn._invalidation_cause = None
    conn._tx_owner = None
    conn._in_transaction = False
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._check_in_use = lambda: None
    conn._ensure_connected = lambda: (conn._protocol, conn._db_id)  # type: ignore[assignment,return-value]
    conn._invalidate = MagicMock()
    return conn


@pytest.mark.asyncio
async def test_cancel_in_baseexceptiongroup_invalidates_connection() -> None:
    """A group containing a ``CancelledError`` must trigger ``_invalidate``."""
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> None:
        raise BaseExceptionGroup(
            "task-group sibling cancel",
            [asyncio.CancelledError(), RuntimeError("sibling failure")],
        )

    with pytest.raises(BaseExceptionGroup):
        await conn._run_protocol(fn)

    assert conn._invalidate.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_non_cancel_baseexceptiongroup_does_not_invalidate() -> None:
    """A pure application-error group must NOT invalidate."""
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> None:
        raise BaseExceptionGroup(
            "two sibling app errors",
            [ValueError("v"), RuntimeError("r")],
        )

    with pytest.raises(BaseExceptionGroup):
        await conn._run_protocol(fn)

    assert conn._invalidate.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_nested_group_with_cancel_invalidates_connection() -> None:
    """A cancel nested in a group-within-a-group must still invalidate; a shallow
    ``any(isinstance(...))`` walk misses it, but ``split()`` recurses (PEP 654)."""
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> None:
        raise BaseExceptionGroup(
            "outer",
            [
                BaseExceptionGroup(
                    "inner",
                    [asyncio.CancelledError()],
                ),
            ],
        )

    with pytest.raises(BaseExceptionGroup):
        await conn._run_protocol(fn)

    assert conn._invalidate.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mixed_nested_group_with_cancel_invalidates_connection() -> None:
    """Outer group with a non-cancel child plus an inner group with a cancel: still invalidates."""
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> None:
        raise BaseExceptionGroup(
            "outer",
            [
                ValueError("app error"),
                BaseExceptionGroup(
                    "inner",
                    [asyncio.CancelledError()],
                ),
            ],
        )

    with pytest.raises(BaseExceptionGroup):
        await conn._run_protocol(fn)

    assert conn._invalidate.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_in_use_is_cleared_on_baseexceptiongroup_propagation() -> None:
    """``_in_use`` must be cleared by ``finally`` regardless of group composition."""
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> None:
        raise BaseExceptionGroup("g", [asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup):
        await conn._run_protocol(fn)

    assert conn._in_use is False


@pytest.mark.asyncio
async def test_real_taskgroup_propagates_invalidating_group() -> None:
    """End-to-end: a sibling failure inside a real TaskGroup invalidates the connection."""
    conn = _make_connection()
    op_started = asyncio.Event()

    async def slow_op(_protocol: object, _db_id: int) -> None:
        op_started.set()
        await asyncio.sleep(10)

    async def sibling() -> None:
        await op_started.wait()
        raise RuntimeError("sibling boom")

    with pytest.raises(BaseExceptionGroup):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(conn._run_protocol(slow_op))
            tg.create_task(sibling())

    assert conn._invalidate.call_count >= 1  # type: ignore[attr-defined]


async def test_fixture_sanity_unwrap_normal_return() -> None:
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> int:
        return 42

    result = await conn._run_protocol(fn)
    assert result == 42
    assert conn._invalidate.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fixture_sanity_async() -> None:
    await test_fixture_sanity_unwrap_normal_return()


_ = AsyncMock  # quiet unused-import lint
