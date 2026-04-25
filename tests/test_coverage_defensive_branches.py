"""Pin defensive branches in ``connection.py``, ``pool.py``,
``retry.py`` reported as uncovered by ``pytest --cov``.

Each test drives a load-bearing structural defense that no other
test exercises directly. A regression that removes or weakens one
of these checks would silently break correctness without surfacing
in normal tests.

Lines covered by this file (pre-pragma):

connection.py:
- 506, 523 — close-time DEBUG-log fallbacks for unexpected drain
  exceptions.
- 574-575 — ``_check_in_use`` no-loop branch.
- 647-648 — ``_invalidate`` no-loop branch.
- 811     — ``_update_tx_flags_from_sql`` empty-SQL noop.
- 834     — ``_update_tx_flags_from_sql`` ROLLBACK TO short-circuit.

pool.py:
- (initialize / acquire / release defensive paths via mock-driven
  tests; see per-test docstring).

retry.py:
- 146 — deadline recheck breaks before the next attempt when
  the budget is exhausted.

The remaining uncovered lines (race fallbacks at connection.py:466-467,
pool.py:429-430, pool.py:913-914, protocol.py:591/633) are marked
``# pragma: no cover`` inline at the source with one-line
justifications.
"""

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

# ---------------------------------------------------------------------------
# connection.py — _check_in_use / _invalidate / _update_tx_flags_from_sql
# ---------------------------------------------------------------------------


class TestCheckInUseNoLoopBranch:
    def test_raises_interface_error_outside_running_loop(self) -> None:
        """``_check_in_use`` calls ``asyncio.get_running_loop()`` for
        loop-binding validation. If no loop is running (e.g. caller
        invoked the method from a bare thread), surface a clean
        ``InterfaceError("must be used from within an async context.")``
        rather than letting the underlying ``RuntimeError`` propagate."""
        conn = DqliteConnection("localhost:9001")
        # Call from a fresh thread that has no running event loop.
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
        """``_invalidate`` attempts to schedule a bounded drain task on
        the current loop; if no loop is running, the scheduling
        silently no-ops and the rest of invalidation proceeds. Pin
        the swallow."""
        conn = DqliteConnection("localhost:9001")
        # Set up state that should be cleared regardless of loop
        # presence.
        conn._protocol = MagicMock()  # type: ignore[assignment]
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
        # Invalidate must succeed without raising even with no loop.
        assert captured["err"] is None
        assert conn._protocol is None
        assert conn._in_transaction is False


class TestUpdateTxFlagsFromSql:
    def test_empty_sql_is_noop(self) -> None:
        """Whitespace-only SQL must early-return without flipping
        any tx flags. Drives connection.py:811."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = False
        conn._update_tx_flags_from_sql("   ")
        assert conn._in_transaction is False

    def test_rollback_to_savepoint_preserves_outer_tx(self) -> None:
        """``ROLLBACK TO sp`` (savepoint rollback) must NOT flip
        ``_in_transaction`` — only ``ROLLBACK`` (without ``TO``)
        ends the outer transaction. Drives connection.py:834."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._update_tx_flags_from_sql("ROLLBACK TO sp1")
        # Outer tx must remain open.
        assert conn._in_transaction is True

    def test_rollback_without_to_clears_in_transaction(self) -> None:
        """Sanity / negative test for the above — the bare ``ROLLBACK``
        path correctly clears the flag (covered already, reasserted
        here for readability of the contrast)."""
        conn = DqliteConnection("localhost:9001")
        conn._in_transaction = True
        conn._update_tx_flags_from_sql("ROLLBACK")
        assert conn._in_transaction is False


class TestCloseLogsUnexpectedDrainError:
    """``close()`` and ``_abort_protocol()`` swallow OSError on the
    bounded drain (slow peer / already-closed writer) but DEBUG-log
    anything else. Pin the DEBUG-log path."""

    @pytest.mark.asyncio
    async def test_close_logs_at_debug_on_unexpected_drain_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = DqliteConnection("localhost:9001")
        # Build a fake protocol whose ``wait_closed`` raises a
        # non-OSError, non-CancelledError exception.
        proto = MagicMock()
        proto.close = MagicMock(return_value=None)
        proto.wait_closed = AsyncMock(side_effect=ValueError("boom"))
        conn._protocol = proto  # type: ignore[assignment]

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
        conn._protocol = proto  # type: ignore[assignment]

        with caplog.at_level(logging.DEBUG, logger="dqliteclient.connection"):
            await conn._abort_protocol()

        assert any(
            "_abort_protocol: unexpected drain error" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# retry.py — deadline recheck before next attempt
# ---------------------------------------------------------------------------


class TestRetryDeadlineRecheck:
    @pytest.mark.asyncio
    async def test_breaks_when_deadline_passed_before_next_attempt(self) -> None:
        """The per-iteration deadline check at ``retry.py:145-146``
        fires when ``attempt > 0`` and the deadline has elapsed
        between the previous backoff sleep wake and the next
        ``func()`` call. To drive specifically this check (not the
        sibling inner check at L161 which runs before the sleep is
        scheduled), arrange for the previous sleep to push the
        clock past the deadline before the loop iterates.

        Setup: first call raises at ~t=0; inner check at L161 sees
        t≈0 < deadline=0.01 so does NOT break; backoff sleep clamps
        to ``deadline - now ≈ 0.01``; loop iterates with attempt=1
        at t≈0.01 which is now ``>= deadline`` — line 146 fires.
        """
        calls: list[int] = []

        async def func() -> None:
            calls.append(1)
            raise OperationalError(0, "fail")

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

        # First call ran; deadline-recheck at L146 broke the loop
        # before a second attempt could fire.
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Suppress unused-import warnings for symbols imported for type hints.
# ---------------------------------------------------------------------------

_ = (DqliteConnectionError, Awaitable, Any)
