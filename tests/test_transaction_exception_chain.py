"""Pin the exception-propagation contracts of ``transaction()``'s
exit path when both the body and the cleanup raise.

The exit path of ``transaction()`` distinguishes three cases:

1. **Body raises + ROLLBACK raises a real OperationalError** (any
   non-no-tx error). The rollback error is CAUGHT and handled
   (logged; connection invalidated). The body exception then
   propagates via bare ``raise``. Per Python's ``__context__``
   semantics, the rollback exception does NOT attach to the body
   exception's chain — it was a separate exception caught and
   handled inside the outer ``except`` block. Operators see only
   the body exception; the rollback failure is in the DEBUG log.

2. **Body raises + ROLLBACK reports the deterministic "no
   transaction is active"**. Same shape as (1) — caught, logged,
   but connection is preserved (server-side tx already gone).
   Body exception propagates with no chain.

3. **Body raises + ROLLBACK is cancelled** (CancelledError /
   KeyboardInterrupt / SystemExit). The cancellation handler
   ``raise``-s the cancellation. Because that ``raise`` is inside
   the outer ``except BaseException``, the body exception attaches
   as ``__context__`` on the cancellation. Cancellation supersedes
   the body exception; connection is invalidated.

Pin all three contracts so a refactor that swaps the bare ``raise``
for ``raise X from Y`` (or accidentally adds chaining where there
isn't any) is caught loudly.
"""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


def _prime_connected(conn: DqliteConnection) -> None:
    """Make a bare DqliteConnection look connected enough for
    ``transaction()`` to run BEGIN/COMMIT/ROLLBACK paths against the
    test's mocked ``execute``."""
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_body_exc_with_real_rollback_failure_invalidates_connection() -> None:
    """Body raises ValueError; ROLLBACK raises a non-no-tx
    OperationalError. The body exception propagates without the
    rollback in its ``__context__`` (caught + handled inside the
    outer except). The connection is invalidated."""
    conn = DqliteConnection("localhost:9001")
    _prime_connected(conn)

    rollback_error = OperationalError(2, "rollback failed for unrelated reason")

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise rollback_error
        return (0, 0)

    conn.execute = fake_execute  # type: ignore[assignment]

    body_exc: ValueError | None = None
    try:
        async with conn.transaction():
            raise ValueError("body failure")
    except ValueError as e:
        body_exc = e

    assert body_exc is not None
    # The rollback OperationalError was caught and handled inside the
    # outer except — it does not attach to body_exc's chain (Python
    # semantics: __context__ is set on the new exception when it is
    # raised, not on a re-raised one). Operators read it from the
    # DEBUG log, not the exception chain.
    assert body_exc.__context__ is None
    # Connection was invalidated because ROLLBACK failed for a
    # non-no-tx reason. `_invalidate()` is called without a cause arg
    # so `_invalidation_cause` stays None; check `_protocol is None`
    # which IS cleared by `_invalidate()`.
    assert conn._protocol is None


@pytest.mark.asyncio
async def test_body_exc_with_no_tx_rollback_preserves_connection() -> None:
    """If ROLLBACK raises with the deterministic "no transaction is
    active" wording, the connection survives. The body exception
    propagates without chain (rollback was caught and handled)."""
    conn = DqliteConnection("localhost:9001")
    _prime_connected(conn)

    no_tx_error = OperationalError(1, "cannot rollback - no transaction is active")

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise no_tx_error
        return (0, 0)

    conn.execute = fake_execute  # type: ignore[assignment]

    body_exc: ValueError | None = None
    try:
        async with conn.transaction():
            raise ValueError("body failure")
    except ValueError as e:
        body_exc = e

    assert body_exc is not None
    assert body_exc.__context__ is None
    # Connection NOT invalidated — no-tx is benign. `_protocol` should
    # remain non-None.
    assert conn._protocol is not None


@pytest.mark.asyncio
async def test_rollback_cancellation_supersedes_body_with_context_chain() -> None:
    """Body raises ValueError; ROLLBACK is cancelled mid-flight. The
    caller sees CancelledError; the body's ValueError attaches via
    ``__context__`` because the cancellation handler does ``raise``
    (re-raising the just-caught cancellation) inside the outer
    ``except BaseException`` scope where body_exc was the active
    exception. Connection is invalidated."""
    conn = DqliteConnection("localhost:9001")
    _prime_connected(conn)

    body_error = ValueError("body failure")

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise asyncio.CancelledError()
        return (0, 0)

    conn.execute = fake_execute  # type: ignore[assignment]

    cancelled_exc: asyncio.CancelledError | None = None
    try:
        async with conn.transaction():
            raise body_error
    except asyncio.CancelledError as e:
        cancelled_exc = e

    assert cancelled_exc is not None
    assert isinstance(cancelled_exc.__context__, ValueError)
    assert cancelled_exc.__context__ is body_error
    # Cancellation mid-rollback invalidates the connection: server-side
    # tx state is unknowable. `_invalidate()` clears `_protocol`.
    assert conn._protocol is None
