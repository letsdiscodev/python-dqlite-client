"""DqliteConnection transactions: exception chaining and rollback failure logging."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from dqliteclient.connection import DqliteConnection
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


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


class TestTransactionRollbackFailureLogged:
    async def test_rollback_failure_emits_debug_log(
        self, conn: DqliteConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def mock_execute(sql: str, params: Any = None) -> tuple[int, int]:
            if "ROLLBACK" in sql:
                raise OSError("connection lost")
            return (0, 0)

        conn.execute = mock_execute

        caplog.set_level(logging.DEBUG, logger="dqliteclient.connection")
        with pytest.raises(ValueError, match="body"):
            async with conn.transaction():
                raise ValueError("body")

        rollback_records = [r for r in caplog.records if "rollback failed" in r.getMessage()]
        assert len(rollback_records) == 1
        rec = rollback_records[0]
        assert "address=localhost:9001" in rec.getMessage()
        assert f"id={id(conn)}" in rec.getMessage()
        assert rec.exc_info is not None

    async def test_rollback_cancellation_emits_debug_log(
        self, conn: DqliteConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def mock_execute(sql: str, params: Any = None) -> tuple[int, int]:
            if "ROLLBACK" in sql:
                raise asyncio.CancelledError
            return (0, 0)

        conn.execute = mock_execute

        caplog.set_level(logging.DEBUG, logger="dqliteclient.connection")
        with pytest.raises(asyncio.CancelledError):
            async with conn.transaction():
                raise ValueError("body")

        cancel_records = [r for r in caplog.records if "rollback was cancelled" in r.getMessage()]
        assert len(cancel_records) == 1
        rec = cancel_records[0]
        assert "address=localhost:9001" in rec.getMessage()
        assert f"id={id(conn)}" in rec.getMessage()
