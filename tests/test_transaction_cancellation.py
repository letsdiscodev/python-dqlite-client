"""Cancellation invariants for ``DqliteConnection.transaction()``."""

from __future__ import annotations

import asyncio
import weakref
from typing import Any
from unittest.mock import MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError


class _FakeProtocol:
    """Controllable DqliteProtocol double; per-verb ``on_*`` hooks default to returning (0, 0)."""

    def __init__(self) -> None:
        self._open = True
        self.log: list[str] = []
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
    conn = DqliteConnection("localhost:9001")
    proto = _FakeProtocol()
    conn._protocol = proto  # type: ignore[assignment]
    conn._db_id = 0
    conn._bound_loop_ref = weakref.ref(asyncio.get_event_loop_policy().get_event_loop())
    return conn, proto


@pytest.fixture
def conn_and_proto() -> Any:
    """Event-loop-aware factory; asyncio tests await it for a fresh pair."""

    async def _make() -> tuple[DqliteConnection, _FakeProtocol]:
        conn = DqliteConnection("localhost:9001")
        proto = _FakeProtocol()
        conn._protocol = proto  # type: ignore[assignment]
        conn._db_id = 0
        conn._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        return conn, proto

    return _make


class TestRollbackFailureInvalidatesConnection:
    """Body raises AND ROLLBACK fails: invalidate so the pool discards the still-open tx."""

    async def test_rollback_failure_invalidates_connection(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        async def _failing_rollback() -> tuple[int, int]:
            raise OperationalError("ROLLBACK failed: connection poisoned", 1)

        proto.on_rollback = _failing_rollback

        with pytest.raises(_BodyError):
            async with conn.transaction():
                raise _BodyError("user error")

        # Invalidate on any ROLLBACK failure; an "or _in_transaction" tolerance is too weak.
        assert conn._protocol is None, (
            "After ROLLBACK failure in body-raised path, connection "
            "must be invalidated (protocol=None) so the pool discards "
            "it instead of recycling a connection with unknown "
            "server-side transaction state."
        )
        assert not conn._in_transaction
        assert conn._tx_owner is None


class TestRollbackHappyPath:
    """Body raises, ROLLBACK succeeds: the connection stays usable."""

    async def test_body_error_with_successful_rollback(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        with pytest.raises(_BodyError):
            async with conn.transaction():
                raise _BodyError()

        assert proto.log == ["BEGIN", "ROLLBACK"]
        assert conn._protocol is proto, "clean rollback must not invalidate"
        assert not conn._in_transaction
        assert conn._tx_owner is None


class TestRollbackCancellationPropagates:
    """CancelledError during the body-error ROLLBACK must propagate, not be swallowed."""

    async def test_cancelled_error_during_rollback_propagates(self, conn_and_proto: Any) -> None:
        conn, proto = await conn_and_proto()

        class _BodyError(Exception):
            pass

        rollback_started = asyncio.Event()

        async def _rollback_that_sleeps() -> tuple[int, int]:
            rollback_started.set()
            await asyncio.sleep(60)
            return (0, 0)

        proto.on_rollback = _rollback_that_sleeps

        async def run_transaction() -> None:
            async with conn.transaction():
                raise _BodyError("user error")

        task = asyncio.create_task(run_transaction())
        await rollback_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestTransactionCancellationPhases:
    """Cancellation-at-phase matrix; each phase clears state and reaches a terminal state."""

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
                pass

        task = asyncio.create_task(run_transaction())
        await begin_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

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
                pass

        task = asyncio.create_task(run_transaction())
        await commit_started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # COMMIT is ambiguous under cancel, so the connection must be invalidated.
        assert conn._protocol is None, (
            "Connection must be invalidated after COMMIT cancellation — "
            "server-side state is ambiguous (maybe committed, maybe not)."
        )
        assert not conn._in_transaction
        assert conn._tx_owner is None

        # Preserve the cancel as invalidation cause so later _ensure_connected raises chain to it.
        assert isinstance(conn._invalidation_cause, asyncio.CancelledError), (
            f"Expected CancelledError as invalidation cause, "
            f"got {type(conn._invalidation_cause).__name__}: "
            f"{conn._invalidation_cause!r}"
        )

    async def test_cancel_during_rollback(self, conn_and_proto: Any) -> None:
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
