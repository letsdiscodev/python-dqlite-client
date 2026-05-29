"""``transaction()`` exit clears savepoint state alongside ``_in_transaction`` / ``_tx_owner``."""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


@pytest.mark.asyncio
async def test_benign_no_tx_rollback_clears_savepoint_state() -> None:
    """Benign no-tx rollback clears the savepoint pair: the server confirms the tx is gone."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    no_tx_error = OperationalError("cannot rollback - no transaction is active", 1)

    async def fake_execute(sql: str, params: object = None) -> tuple[int, int]:
        if sql == "BEGIN":
            conn._savepoint_stack.append("sp")
            conn._savepoint_implicit_begin = True
            return (0, 0)
        if sql == "ROLLBACK":
            raise no_tx_error
        return (0, 0)

    conn.execute = fake_execute

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise RuntimeError("body raised")

    assert conn._invalidation_cause is None
    assert conn._protocol is not None
    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    assert conn._in_transaction is False
    assert conn._tx_owner is None


@pytest.mark.asyncio
async def test_finally_clears_savepoint_state_on_body_exception_with_invalidation() -> None:
    """After non-benign ROLLBACK failure, finally re-clears the pair idempotently."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_execute(sql: str, params: object = None) -> tuple[int, int]:
        if sql == "BEGIN":
            conn._savepoint_stack.append("sp1")
            conn._savepoint_stack.append("sp2")
            conn._savepoint_implicit_begin = True
            return (0, 0)
        if sql == "ROLLBACK":
            raise OperationalError("some other failure mode", 1)
        return (0, 0)

    conn.execute = fake_execute

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise RuntimeError("body raised")

    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False


@pytest.mark.asyncio
async def test_finally_clears_savepoint_state_on_success_path() -> None:
    """Success path: finally enforces the clear even though COMMIT already does it."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_execute(sql: str, params: object = None) -> tuple[int, int]:
        if sql == "BEGIN":
            conn._savepoint_stack.append("sp")
            conn._savepoint_implicit_begin = True
            return (0, 0)
        if sql == "COMMIT":
            return (0, 0)
        return (0, 0)

    conn.execute = fake_execute

    async with conn.transaction():
        pass

    assert conn._savepoint_stack == []
    assert conn._savepoint_implicit_begin is False
    assert conn._in_transaction is False
    assert conn._tx_owner is None
