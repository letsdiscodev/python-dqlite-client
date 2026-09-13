"""Connection contract on a stubbed protocol: transaction tracking, invalidation, transaction()."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import dqliteclient
from dqliteclient import (
    AmbiguousCommitError,
    DqliteConnection,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqlitewire import LEADER_ERROR_CODES, SQLITE_IOERR_LEADERSHIP_LOST
from dqlitewire.constants import TX_AUTO_ROLLBACK_PRIMARY_CODES


@pytest.mark.parametrize(
    ("statements", "expected"),
    [
        (["BEGIN"], True),
        (["BEGIN", "COMMIT"], False),
        (["BEGIN", "END"], False),
        (["begin immediate", "rollback"], False),
        (["SAVEPOINT a"], True),
        (["SAVEPOINT a", "RELEASE a"], True),
        (["SAVEPOINT a", "ROLLBACK TO a"], True),
        (["SAVEPOINT a", "ROLLBACK TRANSACTION TO SAVEPOINT a"], True),
        (["SAVEPOINT a", "ROLLBACK"], False),
        (["/* c */ BEGIN; INSERT INTO t VALUES (1); COMMIT"], False),
        (["BEGIN; INSERT INTO t VALUES (1)"], True),
        (["INSERT INTO t VALUES (1)"], False),
    ],
)
async def test_transaction_flag_follows_the_tracking_table(
    connected: Any, statements: list[str], expected: bool
) -> None:
    conn, _ = connected()
    for sql in statements:
        await conn.execute(sql)
    assert conn.in_transaction is expected


async def test_no_transaction_reply_clears_flag_without_invalidating(connected: Any) -> None:
    conn, proto = connected()
    await conn.execute("SAVEPOINT a")
    await conn.execute("RELEASE a")
    proto.fail_with["ROLLBACK"] = OperationalError(
        "cannot rollback - no transaction is active", code=1
    )
    with pytest.raises(OperationalError):
        await conn.execute("ROLLBACK")
    assert conn.in_transaction is False
    assert conn.is_connected


async def test_auto_rollback_code_clears_flag(connected: Any) -> None:
    conn, proto = connected()
    await conn.execute("BEGIN")
    proto.fail_with["INSERT"] = OperationalError(
        "full", code=next(iter(TX_AUTO_ROLLBACK_PRIMARY_CODES))
    )
    with pytest.raises(OperationalError):
        await conn.execute("INSERT INTO t VALUES (1)")
    assert conn.in_transaction is False
    assert conn.is_connected


async def test_failed_statement_leaves_flag_alone(connected: Any) -> None:
    conn, proto = connected()
    await conn.execute("BEGIN")
    proto.fail_with["INSERT"] = OperationalError("constraint", code=19)
    with pytest.raises(OperationalError):
        await conn.execute("INSERT INTO t VALUES (1)")
    assert conn.in_transaction is True


@pytest.mark.parametrize(
    "exc",
    [
        OperationalError("not leader", code=next(iter(LEADER_ERROR_CODES))),
        DqliteConnectionError("Connection closed by server"),
        ProtocolError("bad frame"),
        asyncio.CancelledError(),
    ],
)
async def test_lost_session_invalidates_the_connection(connected: Any, exc: BaseException) -> None:
    conn, proto = connected()
    await conn.execute("BEGIN")
    proto.fail_with["INSERT"] = exc
    with pytest.raises(type(exc)):
        await conn.execute("INSERT INTO t VALUES (1)")
    assert not conn.is_connected
    assert conn.in_transaction is False
    assert not conn.closed
    with pytest.raises(DqliteConnectionError, match="Not connected"):
        await conn.execute("SELECT 1")


async def test_transaction_context_manager_commits_and_rolls_back(connected: Any) -> None:
    conn, proto = connected()
    async with conn.transaction():
        await conn.execute("INSERT INTO t VALUES (1)")
    assert proto.sent == ["BEGIN", "INSERT INTO t VALUES (1)", "COMMIT"]
    assert conn.in_transaction is False
    with pytest.raises(ValueError, match="body"):
        async with conn.transaction():
            raise ValueError("body")
    assert proto.sent[-2:] == ["BEGIN", "ROLLBACK"]
    assert conn.in_transaction is False


async def test_transaction_rejects_nesting(connected: Any) -> None:
    conn, _ = connected()
    await conn.execute("BEGIN")
    with pytest.raises(InterfaceError, match="Nested transactions"):
        async with conn.transaction():
            pass


async def test_commit_during_leadership_loss_is_ambiguous(connected: Any) -> None:
    conn, proto = connected()
    proto.fail_with["COMMIT"] = OperationalError(
        "leadership lost", code=SQLITE_IOERR_LEADERSHIP_LOST
    )
    with pytest.raises(AmbiguousCommitError):
        async with conn.transaction():
            pass


async def test_concurrent_use_is_rejected(connected: Any, fake_protocol: Any) -> None:
    started = asyncio.Event()

    class SlowProtocol(fake_protocol):  # type: ignore[misc]
        async def exec_sql(self, db_id: int, sql: str, params: Any) -> tuple[int, int]:
            started.set()
            await asyncio.sleep(0.05)
            return (0, 0)

    conn, _ = connected(SlowProtocol())
    task = asyncio.create_task(conn.execute("SELECT 1"))
    await started.wait()
    with pytest.raises(InterfaceError, match="another operation is in progress"):
        await conn.execute("SELECT 2")
    await task


async def test_close_is_idempotent_and_final(connected: Any) -> None:
    conn, proto = connected()
    await conn.close()
    await conn.close()
    assert conn.closed and not conn.is_connected and not proto.is_alive
    with pytest.raises(InterfaceError, match="closed"):
        await conn.connect()


async def test_module_connect_terminates_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[DqliteConnection] = []
    original_init = DqliteConnection.__init__

    def tracking_init(self: DqliteConnection, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        created.append(self)

    async def failing_connect(self: DqliteConnection) -> None:
        raise OSError("boom")

    monkeypatch.setattr(DqliteConnection, "__init__", tracking_init)
    monkeypatch.setattr(DqliteConnection, "connect", failing_connect)
    with pytest.raises(OSError, match="boom"):
        await dqliteclient.connect("localhost:9001")
    assert len(created) == 1 and created[0].closed
