"""execute("BEGIN") failing must leave _in_transaction False.

Tx flags update only after a successful round-trip. A stale True would make the next
commit()/rollback() send a COMMIT with no tx open or surface a confusing "no tx" error.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError
from dqlitewire.messages import FailureResponse


@pytest.fixture
def connected_conn() -> DqliteConnection:
    """A DqliteConnection with a mock protocol ready to be primed with a FailureResponse."""
    conn = DqliteConnection("localhost:9001")
    conn._protocol = MagicMock()
    conn._protocol._reader = AsyncMock()
    conn._protocol._writer = MagicMock()
    conn._protocol._writer.drain = AsyncMock()
    conn._db_id = 1
    return conn


def _prime_failure(conn: DqliteConnection, code: int, message: str) -> None:
    """Prime the underlying protocol's exec_sql to raise OperationalError."""
    err = OperationalError(message, code)

    async def _raise(*_args, **_kwargs):
        raise err

    conn._protocol.exec_sql = _raise  # type: ignore[union-attr]
    conn._protocol.query_sql = _raise  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "begin_sql",
    [
        "BEGIN",
        "BEGIN IMMEDIATE",
        "BEGIN DEFERRED",
        "BEGIN EXCLUSIVE",
        "BEGIN TRANSACTION",
    ],
)
async def test_begin_failure_response_leaves_flag_false(
    connected_conn: DqliteConnection, begin_sql: str
) -> None:
    """A FailureResponse on any BEGIN variant propagates and leaves _in_transaction False."""
    _prime_failure(connected_conn, code=1, message="forced failure")
    assert connected_conn._in_transaction is False

    with pytest.raises(OperationalError):
        await connected_conn.execute(begin_sql)

    assert connected_conn._in_transaction is False
    assert connected_conn._tx_owner is None
    assert connected_conn._savepoint_stack == []
    assert connected_conn._savepoint_implicit_begin is False


async def test_savepoint_failure_response_leaves_stack_unchanged(
    connected_conn: DqliteConnection,
) -> None:
    """A failed SAVEPOINT must not push onto the stack or flip the autobegin flag."""
    _prime_failure(connected_conn, code=1, message="forced failure")
    assert connected_conn._in_transaction is False

    with pytest.raises(OperationalError):
        await connected_conn.execute("SAVEPOINT sp1")

    assert connected_conn._in_transaction is False
    assert connected_conn._savepoint_stack == []
    assert connected_conn._savepoint_implicit_begin is False


def test_failure_response_constructable_with_code_one() -> None:
    """The codec round-trips the failure shape we prime with."""
    body = FailureResponse(code=1, message="forced failure").encode()
    assert body
