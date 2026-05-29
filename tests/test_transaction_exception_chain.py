"""Exception-propagation contracts of ``transaction()``'s exit when body and cleanup both raise:
a caught/handled ROLLBACK error does NOT chain onto the body exception, but a cancelled ROLLBACK
re-raises and the body attaches as ``__context__``."""

from __future__ import annotations

import asyncio

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


def _prime_connected(conn: DqliteConnection) -> None:
    """Make a bare connection look connected enough to run BEGIN/COMMIT/ROLLBACK paths."""
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_body_exc_with_real_rollback_failure_invalidates_connection() -> None:
    """Body raises, ROLLBACK errors: body propagates without chain; conn invalidated."""
    conn = DqliteConnection("localhost:9001")
    _prime_connected(conn)

    rollback_error = OperationalError("rollback failed for unrelated reason", 2)

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise rollback_error
        return (0, 0)

    conn.execute = fake_execute

    body_exc: ValueError | None = None
    try:
        async with conn.transaction():
            raise ValueError("body failure")
    except ValueError as e:
        body_exc = e

    assert body_exc is not None
    # Rollback error was caught/handled, so it does not attach to body_exc's chain.
    assert body_exc.__context__ is None
    # _invalidate() is called without a cause, so check _protocol (cleared), not the cause.
    assert conn._protocol is None


@pytest.mark.asyncio
async def test_body_exc_with_no_tx_rollback_preserves_connection() -> None:
    """Benign no-tx ROLLBACK error: connection survives; body propagates without chain."""
    conn = DqliteConnection("localhost:9001")
    _prime_connected(conn)

    no_tx_error = OperationalError("cannot rollback - no transaction is active", 1)

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise no_tx_error
        return (0, 0)

    conn.execute = fake_execute

    body_exc: ValueError | None = None
    try:
        async with conn.transaction():
            raise ValueError("body failure")
    except ValueError as e:
        body_exc = e

    assert body_exc is not None
    assert body_exc.__context__ is None
    assert conn._protocol is not None


@pytest.mark.asyncio
async def test_rollback_cancellation_supersedes_body_with_context_chain() -> None:
    """ROLLBACK cancelled mid-flight: caller sees CancelledError with body as __context__."""
    conn = DqliteConnection("localhost:9001")
    _prime_connected(conn)

    body_error = ValueError("body failure")

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise asyncio.CancelledError()
        return (0, 0)

    conn.execute = fake_execute

    cancelled_exc: asyncio.CancelledError | None = None
    try:
        async with conn.transaction():
            raise body_error
    except asyncio.CancelledError as e:
        cancelled_exc = e

    assert cancelled_exc is not None
    assert isinstance(cancelled_exc.__context__, ValueError)
    assert cancelled_exc.__context__ is body_error
    assert conn._protocol is None
