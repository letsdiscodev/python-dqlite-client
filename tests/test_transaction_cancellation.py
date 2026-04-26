"""Transaction-level cancellation invariants.

Pins the cancellation behavior of ``DqliteConnection.transaction()``:

- ROLLBACK failure in the body-raised path invalidates the connection
  so the pool discards it on return. Without invalidation, the pool
  would recycle a connection whose server-side transaction is still
  open.
- ``transaction()`` must not swallow ``CancelledError`` raised during
  ROLLBACK. Structured concurrency (``TaskGroup`` / ``asyncio.timeout()``)
  relies on cancellation propagating.
- Parameterized cancellation-at-phase matrix:
    - ``before_begin``: cancel between ``transaction()`` entry and BEGIN
      returning
    - ``in_body``: cancel during the yield body
    - ``during_commit``: cancel during COMMIT
    - ``during_rollback``: cancel during ROLLBACK (after body raised)

Each test exercises one invariant against a controllable protocol fake.
No cluster required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError


class _FakeProtocol:
    """Controllable DqliteProtocol double.

    Per-SQL-verb hooks: ``on[verb]`` may be set to a callable returning
    an awaitable. Default behavior: each verb returns (0, 0) immediately.
    The fake also exposes ``is_closing`` / ``at_eof`` for the pool's
    socket-liveness probe via ``_writer.transport`` and ``_reader``.
    """

    def __init__(self) -> None:
        self._open = True
        self.log: list[str] = []
        # One hook per SQL verb; callable returns awaitable or raises.
        self.on_begin: Any = self._default_ok
        self.on_commit: Any = self._default_ok
        self.on_rollback: Any = self._default_ok
        # Minimal transport/reader shape to satisfy pool's liveness probe.
        self._writer = MagicMock()
        self._writer.transport = MagicMock()
        self._writer.transport.is_closing = lambda: not self._open
        self._reader = MagicMock()
        self._reader.at_eof = lambda: not self._open

    async def _default_ok(self) -> tuple[int, int]:
        return (0, 0)

    async def exec_sql(self, db_id: int, sql: str, params: Any = None) -> tuple[int, int]:
        verb = sql.strip().upper().split()[0]
        self.log.append(verb)
        hook = {
            "BEGIN": self.on_begin,
            "COMMIT": self.on_commit,
            "ROLLBACK": self.on_rollback,
        }.get(verb, self._default_ok)
        return await hook()  # type: ignore[no-any-return]

    def close(self) -> None:
        self._open = False

    async def wait_closed(self) -> None:
        return None


def _make_connection() -> tuple[DqliteConnection, _FakeProtocol]:
    """Build a DqliteConnection wired up to a fake protocol, no TCP."""
    conn = DqliteConnection("localhost:9001")
    proto = _FakeProtocol()
    conn._protocol = proto  # type: ignore[assignment]
    conn._db_id = 0
    conn._bound_loop = asyncio.get_event_loop_policy().get_event_loop()
    return conn, proto


@pytest.fixture
def conn_and_proto() -> Any:
    """Event-loop-aware fixture; asyncio test functions receive a fresh pair."""

    async def _make() -> tuple[DqliteConnection, _FakeProtocol]:
        conn = DqliteConnection("localhost:9001")
        proto = _FakeProtocol()
        conn._protocol = proto  # type: ignore[assignment]
        conn._db_id = 0
        conn._bound_loop = asyncio.get_running_loop()
        return conn, proto

    return _make


class TestRollbackFailureInvalidatesConnection:
    """When the body raises AND ROLLBACK also fails, the connection
    must be marked invalid so the pool discards it instead of
    recycling a Python-side ``_in_transaction=False`` connection that
    has a live server-side transaction.
    """

    async def test_rollback_failure_invalidates_connection(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        # Body will raise. ROLLBACK will also raise.
        async def _failing_rollback() -> tuple[int, int]:
            raise OperationalError(1, "ROLLBACK failed: connection poisoned")

        proto.on_rollback = _failing_rollback

        with pytest.raises(_BodyError):
            async with conn.transaction():
                raise _BodyError("user error")

        # Strict invariant: the fix invalidates the connection when
        # ROLLBACK fails (any reason, cancellation or error). The
        # previous "or _in_transaction" tolerance was too weak — it
        # would accept a still-open server-side transaction with a
        # Python flag that the pool might not inspect.
        assert conn._protocol is None, (
            "After ROLLBACK failure in body-raised path, connection "
            "must be invalidated (protocol=None) so the pool discards "
            "it instead of recycling a connection with unknown "
            "server-side transaction state."
        )
        # Finally block must still clear transaction state.
        assert not conn._in_transaction
        assert conn._tx_owner is None


class TestRollbackHappyPath:
    """Regression: body raises, ROLLBACK succeeds. Verify the fix did
    not break the normal error-recovery path."""

    async def test_body_error_with_successful_rollback(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        with pytest.raises(_BodyError):
            async with conn.transaction():
                raise _BodyError()

        # ROLLBACK must have been sent; state must be clean; connection
        # must be reusable (NOT invalidated — clean rollback preserves
        # the connection).
        assert proto.log == ["BEGIN", "ROLLBACK"]
        assert conn._protocol is proto, "clean rollback must not invalidate"
        assert not conn._in_transaction
        assert conn._tx_owner is None


class TestRollbackCancellationPropagates:
    """CancelledError raised during the body-error ROLLBACK path must
    not be swallowed. ``suppress(BaseException)`` catches it and the
    enclosing task continues as if never cancelled, breaking
    structured-concurrency guarantees.
    """

    async def test_cancelled_error_during_rollback_propagates(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        rollback_started = asyncio.Event()

        async def _rollback_that_sleeps() -> tuple[int, int]:
            rollback_started.set()
            await asyncio.sleep(60)  # will be cancelled
            return (0, 0)

        proto.on_rollback = _rollback_that_sleeps

        async def run_transaction() -> None:
            async with conn.transaction():
                raise _BodyError("user error")

        task = asyncio.create_task(run_transaction())
        await rollback_started.wait()
        task.cancel()

        # After cancellation, the task must be either CancelledError (the
        # new, fixed behavior) or _BodyError (the current bug — cancel was
        # swallowed). We assert the fixed behavior.
        with pytest.raises(asyncio.CancelledError):
            await task


class TestTransactionCancellationPhases:
    """Exhaustive cancellation-at-phase matrix.

    For each phase, we assert:
    1. CancelledError is observed by the caller.
    2. ``_in_transaction`` is cleared (no dangling Python-side state).
    3. The connection is in a terminal state: either cleanly usable
       (rolled back) or invalidated — never "looks clean but server
       has open tx".
    """

    async def test_cancel_before_begin(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        begin_started = asyncio.Event()

        async def _begin_that_blocks() -> tuple[int, int]:
            begin_started.set()
            await asyncio.sleep(60)
            return (0, 0)

        proto.on_begin = _begin_that_blocks

        async def run_transaction() -> None:
            async with conn.transaction():
                pass  # unreachable

        task = asyncio.create_task(run_transaction())
        await begin_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Post-conditions.
        assert not conn._in_transaction, "in_transaction must be cleared"
        assert conn._tx_owner is None, "tx_owner must be cleared"

    async def test_cancel_in_body(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        body_entered = asyncio.Event()

        async def run_transaction() -> None:
            async with conn.transaction():
                body_entered.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(run_transaction())
        await body_entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # ROLLBACK should have been attempted; state must be clean.
        assert "ROLLBACK" in proto.log, "ROLLBACK must be attempted on cancel-in-body"
        assert not conn._in_transaction
        assert conn._tx_owner is None

    async def test_cancel_during_commit(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        commit_started = asyncio.Event()

        async def _commit_that_blocks() -> tuple[int, int]:
            commit_started.set()
            await asyncio.sleep(60)
            return (0, 0)

        proto.on_commit = _commit_that_blocks

        async def run_transaction() -> None:
            async with conn.transaction():
                pass  # yield succeeds immediately

        task = asyncio.create_task(run_transaction())
        await commit_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Connection must be invalidated (COMMIT is ambiguous under cancel).
        # Same contract as the commit_attempted branch above.
        assert conn._protocol is None, (
            "Connection must be invalidated after COMMIT cancellation — "
            "server-side state is ambiguous (maybe committed, maybe not)."
        )
        assert not conn._in_transaction
        assert conn._tx_owner is None

        # The cancel that triggered the invalidation must be preserved
        # as the invalidation cause so subsequent ``_ensure_connected``
        # raises chain back to it. Operators reading "Not connected"
        # logs need a breadcrumb to the originating cancel.
        assert isinstance(conn._invalidation_cause, asyncio.CancelledError), (
            f"Expected CancelledError as invalidation cause, "
            f"got {type(conn._invalidation_cause).__name__}: "
            f"{conn._invalidation_cause!r}"
        )

    async def test_cancel_during_rollback(self, conn_and_proto: Any) -> None:
        # Body-error ROLLBACK cancellation, framed as a phase test.
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        rollback_started = asyncio.Event()

        async def _rollback_that_blocks() -> tuple[int, int]:
            rollback_started.set()
            await asyncio.sleep(60)
            return (0, 0)

        proto.on_rollback = _rollback_that_blocks

        async def run_transaction() -> None:
            async with conn.transaction():
                raise _BodyError()

        task = asyncio.create_task(run_transaction())
        await rollback_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert not conn._in_transaction
        assert conn._tx_owner is None
