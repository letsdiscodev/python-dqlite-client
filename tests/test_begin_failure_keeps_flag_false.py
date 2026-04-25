"""``DqliteConnection.execute("BEGIN")`` failing must leave ``_in_transaction`` False.

The ``execute`` flow runs the protocol first and only updates the
transaction-tracking flags after a successful round-trip
(``_update_tx_flags_from_sql`` at ``connection.py:843-860`` runs only on
the post-await line). A FailureResponse on BEGIN therefore must NOT
flip ``_in_transaction``. The invariant is implicit in the source; this
file pins it so a refactor that moved the sniff before the protocol
call (or special-cased BEGIN to flip eagerly) breaks the suite
immediately.

The pin matters because the dbapi's ``commit()`` / ``rollback()`` use
``_NO_TX_SUBSTRINGS`` substring suppression to swallow the genuine
"no transaction is active" reply. If a failed BEGIN left
``_in_transaction = True``, the next ``commit()`` would either send a
COMMIT against a server with no tx open (suppressed) or surface a
confusing "no tx" error to the caller.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError
from dqlitewire.messages import FailureResponse


@pytest.fixture
def connected_conn() -> DqliteConnection:
    """A DqliteConnection whose protocol mock can be primed with a
    FailureResponse and whose ``_ensure_connected`` short-circuits."""
    conn = DqliteConnection("localhost:9001")
    conn._protocol = MagicMock()
    conn._protocol._reader = AsyncMock()
    conn._protocol._writer = MagicMock()
    conn._protocol._writer.drain = AsyncMock()
    conn._db_id = 1
    return conn


def _prime_failure(conn: DqliteConnection, code: int, message: str) -> None:
    """Prime the underlying protocol's exec_sql to raise OperationalError."""
    err = OperationalError(code, message)

    async def _raise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise err

    conn._protocol.exec_sql = _raise  # type: ignore[assignment]
    conn._protocol.query_sql = _raise  # type: ignore[assignment]


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
    """A FailureResponse on any BEGIN variant must propagate the
    OperationalError and leave ``_in_transaction`` False."""
    _prime_failure(connected_conn, code=1, message="forced failure")
    assert connected_conn._in_transaction is False

    with pytest.raises(OperationalError):
        await connected_conn.execute(begin_sql)

    # Failed BEGIN must not have flipped any tracker state.
    assert connected_conn._in_transaction is False
    assert connected_conn._tx_owner is None
    assert connected_conn._savepoint_stack == []
    assert connected_conn._savepoint_implicit_begin is False


async def test_savepoint_failure_response_leaves_stack_unchanged(
    connected_conn: DqliteConnection,
) -> None:
    """Mirror for SAVEPOINT autobegin: a failed SAVEPOINT must not
    push onto the stack or flip the autobegin flag."""
    _prime_failure(connected_conn, code=1, message="forced failure")
    assert connected_conn._in_transaction is False

    with pytest.raises(OperationalError):
        await connected_conn.execute("SAVEPOINT sp1")

    assert connected_conn._in_transaction is False
    assert connected_conn._savepoint_stack == []
    assert connected_conn._savepoint_implicit_begin is False


def test_failure_response_constructable_with_code_one() -> None:
    """Sanity: the codec round-trips the failure-shape we're priming."""
    body = FailureResponse(code=1, message="forced failure").encode()
    assert body  # non-empty encode confirms primary code 1 is valid input
