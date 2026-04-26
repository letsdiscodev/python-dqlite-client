"""Pin: ``transaction()`` ROLLBACK-failure paths emit DEBUG log records.

Operators correlating production logs across the dbapi and client
layers need to disambiguate "body exception" from "rollback failure"
when both surface during a transaction. The dbapi layer's ``__exit__``
already logs at DEBUG (ISSUE-305 / ISSUE-469 done); the client-layer
``transaction()`` was the missing peer.

Pin two records (one per branch):
- ROLLBACK failed for a non-cancellation reason: substring
  ``"rollback failed"``.
- ROLLBACK was cancelled mid-flight: substring
  ``"rollback was cancelled"``.

Both records carry ``(address=..., id=...)`` correlator tokens matching
the dbapi-layer convention.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from dqliteclient.connection import DqliteConnection


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

        # Exactly one record from this path.
        rollback_records = [r for r in caplog.records if "rollback failed" in r.getMessage()]
        assert len(rollback_records) == 1
        rec = rollback_records[0]
        assert "address=localhost:9001" in rec.getMessage()
        assert f"id={id(conn)}" in rec.getMessage()
        # ``exc_info=True`` should populate exception details on the record.
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
