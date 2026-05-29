"""Defensive branches in connection.py, pool.py, retry.py that no other test
exercises directly."""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import (
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
)
from dqliteclient.retry import retry_with_backoff


class TestCheckInUseNoLoopBranch:
    def test_raises_interface_error_outside_running_loop(self) -> None:
        """No running loop must surface as InterfaceError, not a raw RuntimeError."""
        conn = DqliteConnection("localhost:9001")
        captured: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                conn._check_in_use()
            except BaseException as e:  # noqa: BLE001
                captured["err"] = e

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        err = captured.get("err")
        assert isinstance(err, InterfaceError)
        assert "async context" in str(err)


class TestInvalidateNoLoopBranch:
    def test_invalidate_swallows_no_running_loop(self) -> None:
        """With no running loop, drain scheduling no-ops and invalidation still proceeds."""
        conn = DqliteConnection("localhost:9001")
        conn._protocol = MagicMock()
        conn._in_transaction = True

        captured: dict[str, BaseException | None] = {"err": None}

        def _runner() -> None:
            try:
                conn._invalidate()
            except BaseException as e:  # noqa: BLE001
                captured["err"] = e

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        assert captured["err"] is None
        assert conn._protocol is None
        assert conn._in_transaction is False


class TestUpdateTxFlagsFromSql:
    def test_empty_sql_is_noop(self) -> None:
        """Whitespace-only SQL early-returns without flipping tx flags."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = False
        conn._update_tx_flags_from_sql("   ")
        assert conn._in_transaction is False

    def test_rollback_to_savepoint_preserves_outer_tx(self) -> None:
        """ROLLBACK TO (savepoint) must NOT end the outer tx; only bare ROLLBACK does."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._update_tx_flags_from_sql("ROLLBACK TO sp1")
        assert conn._in_transaction is True

    def test_rollback_without_to_clears_in_transaction(self) -> None:
        """Contrast: bare ROLLBACK clears the flag."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._in_transaction is False


class TestCloseLogsUnexpectedDrainError:
    """close()/_abort_protocol() swallow OSError on the bounded drain but
    DEBUG-log anything else."""

    @pytest.mark.asyncio
    async def test_close_logs_at_debug_on_unexpected_drain_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = DqliteConnection("localhost:9001")
        # wait_closed raises a non-OSError, non-CancelledError exception.
        proto = MagicMock()
        proto.close = MagicMock(return_value=None)
        proto.wait_closed = AsyncMock(side_effect=ValueError("boom"))
        conn._protocol = proto

        with caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"):
            await conn.close()

        assert any("close: unexpected drain error" in rec.message for rec in caplog.records), (
            f"expected DEBUG record; got {[r.message for r in caplog.records]!r}"
        )

    @pytest.mark.asyncio
    async def test_abort_protocol_logs_at_debug_on_unexpected_drain_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = DqliteConnection("localhost:9001")
        proto = MagicMock()
        proto.close = MagicMock(return_value=None)
        proto.wait_closed = AsyncMock(side_effect=ValueError("boom"))
        conn._protocol = proto

        with caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"):
            await conn._abort_protocol()

        assert any(
            "_abort_protocol: unexpected drain error" in rec.message for rec in caplog.records
        )


class TestRetryDeadlineRecheck:
    @pytest.mark.asyncio
    async def test_breaks_when_deadline_passed_before_next_attempt(self) -> None:
        """The per-iteration deadline recheck breaks before the next attempt when
        the backoff sleep pushed the clock past the deadline."""
        calls: list[int] = []

        async def func() -> None:
            calls.append(1)
            raise OperationalError("fail", 0)

        with pytest.raises(OperationalError):
            await retry_with_backoff(
                func,
                max_attempts=3,
                base_delay=0.05,  # large enough to push past deadline post-sleep
                max_delay=0.05,
                jitter=0.0,
                max_elapsed_seconds=0.01,
                retryable_exceptions=(OperationalError,),
            )

        # Deadline recheck broke the loop before a second attempt.
        assert len(calls) == 1


# Suppress unused-import warnings for symbols imported for type hints.
_ = (DqliteConnectionError, Awaitable, Any)
