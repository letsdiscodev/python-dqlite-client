"""A "no transaction is active" reply on ROLLBACK must not invalidate
the connection (client-layer mirror of the dbapi _NO_TX_CODES whitelist)."""

from __future__ import annotations

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import InterfaceError, OperationalError


@pytest.mark.asyncio
async def test_rollback_no_transaction_active_preserves_connection() -> None:
    """A no-transaction OperationalError on ROLLBACK must leave the
    connection healthy — only the body exception propagates."""
    conn = DqliteConnection("localhost:9001")

    # Pretend the connection is open so transaction() takes the SQL paths.
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    body_error = RuntimeError("body raised")
    no_tx_error = OperationalError(
        "cannot rollback - no transaction is active",
        1,  # SQLITE_ERROR
    )

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise no_tx_error
        return (0, 0)

    conn.execute = fake_execute

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise body_error

    assert conn._invalidation_cause is None
    assert conn._protocol is not None
    assert conn._in_transaction is False
    assert conn._tx_owner is None


@pytest.mark.asyncio
async def test_rollback_other_operational_error_invalidates() -> None:
    """A non-no-tx OperationalError on ROLLBACK still invalidates."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    other_error = OperationalError(
        "some other failure mode",
        1,  # SQLITE_ERROR
    )

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise other_error
        return (0, 0)

    conn.execute = fake_execute

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise RuntimeError("body raised")

    assert conn._protocol is None


@pytest.mark.asyncio
async def test_rollback_non_operational_error_invalidates() -> None:
    """A non-OperationalError on ROLLBACK still invalidates."""
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]

    async def fake_execute(sql: str, params=None):
        if sql == "BEGIN":
            return (0, 0)
        if sql == "ROLLBACK":
            raise InterfaceError("interface broke mid-rollback")
        return (0, 0)

    conn.execute = fake_execute

    with pytest.raises(RuntimeError, match="body raised"):
        async with conn.transaction():
            raise RuntimeError("body raised")

    assert conn._protocol is None
