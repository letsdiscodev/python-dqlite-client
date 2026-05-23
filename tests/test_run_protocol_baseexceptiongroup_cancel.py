"""Pin: ``DqliteConnection._run_protocol`` invalidates the connection
when a ``BaseExceptionGroup`` containing a cancel-class child
propagates from a structured-concurrency primitive
(``asyncio.TaskGroup``, ``anyio.create_task_group``, ``trio``).

PEP 654 says ``isinstance(eg, CancelledError)`` is False for an
``asyncio.TaskGroup``-wrapped sibling cancel. Without an explicit
``except BaseExceptionGroup`` arm, the group would propagate
uninvalidated through ``_run_protocol`` and the connection's wire
state would be silently reused on the next op — breaking the
cancel-invalidate invariant the bare-class arm documents.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection


def _make_connection() -> DqliteConnection:
    """Construct a DqliteConnection with the minimum state to exercise
    ``_run_protocol``'s except chain. The ``_check_in_use`` guard is
    bypassed; the test focuses on the cancel-class arm only."""
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
    # Bypass the in-use / fork / pool / loop guards so the test
    # exercises only the except chain.
    conn._check_in_use = lambda: None
    conn._ensure_connected = lambda: (conn._protocol, conn._db_id)  # type: ignore[assignment,return-value]
    conn._invalidate = MagicMock()
    return conn


@pytest.mark.asyncio
async def test_cancel_in_baseexceptiongroup_invalidates_connection() -> None:
    """A ``BaseExceptionGroup`` whose children include a
    ``CancelledError`` must trigger ``_invalidate`` — the bare-class
    arm's invariant extends to grouped form (PEP 654)."""
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
    """A group with no cancel-class child uses inner-arm policy; the
    new BaseExceptionGroup arm must NOT invalidate on a pure
    application-error group (the per-class arms above already encode
    the precise policy for those classes)."""
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
    """A ``BaseExceptionGroup`` nested inside another
    ``BaseExceptionGroup`` (the shape ``asyncio.TaskGroup`` produces
    when an inner TaskGroup propagates out through an outer one) must
    still trigger ``_invalidate``. A shallow ``any(isinstance(child,
    cancel_classes))`` walk misses the nested cancel; the PEP 654
    idiom ``BaseExceptionGroup.split()`` recurses by design and is the
    correct primitive."""
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
    """Mirror the production shape where the outer group carries both
    a non-cancel child (sibling app error) and an inner group with a
    cancel — the cancel still has to surface to ``_invalidate``."""
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
    """Sibling invariant: regardless of group composition, ``_in_use``
    must be cleared by the ``finally`` block so the connection is not
    locked out for future ops (or for its eventual close)."""
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> None:
        raise BaseExceptionGroup("g", [asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup):
        await conn._run_protocol(fn)

    assert conn._in_use is False


@pytest.mark.asyncio
async def test_real_taskgroup_propagates_invalidating_group() -> None:
    """End-to-end: drive ``_run_protocol`` from inside a TaskGroup
    body where a sibling task fails. The resulting
    ``BaseExceptionGroup`` invalidates the connection."""
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


# Drive the AsyncMock-based fn through _run_protocol so the
# in_use guard is exercised once on each test (sanity for the
# fixture).
async def test_fixture_sanity_unwrap_normal_return() -> None:
    conn = _make_connection()

    async def fn(_protocol: object, _db_id: int) -> int:
        return 42

    result = await conn._run_protocol(fn)
    assert result == 42
    assert conn._invalidate.call_count == 0  # type: ignore[attr-defined]


# Re-export so pytest-asyncio binds the helper async function.
@pytest.mark.asyncio
async def test_fixture_sanity_async() -> None:
    await test_fixture_sanity_unwrap_normal_return()


# Quiet AsyncMock lint warning about unused symbol.
_ = AsyncMock
