"""Pin: multi-statement EXEC with mid-batch failure sets the
conservative ``_has_untracked_savepoint`` flag so pool reset's
safety ROLLBACK fires.

Round 2's ``done/savepoint-multi-statement-execute-tracker-desync.md``
fixed the success path: ``_update_tx_flags_from_sql`` splits on
top-level ``;`` and recurses, so ``BEGIN; SAVEPOINT a;`` correctly
tracks both verbs. The partial-failure converse — where an early
statement (BEGIN, SAVEPOINT) commits server-side state before a
later statement (INSERT) fails — was unaddressed: the success-only
``_update_tx_flags_from_sql`` call in ``execute()`` is skipped on
raise, leaving the local view out of sync with the server's open
transaction.

The fix is the conservative ``_has_untracked_savepoint`` flag:
on EXEC failure, if the SQL contains ``;`` AND any tx-control verb
appears in the split pieces, set the flag. Pool reset's safety
ROLLBACK then fires (via the existing predicate that ORs the flag).

False-positive ROLLBACK on benign multi-INSERT batches is acceptable
— they don't run multi-statement tx-control verbs, so the flag
predicate doesn't fire.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dqliteclient import DqliteConnection
from dqliteclient.exceptions import OperationalError


async def _raise_constraint(_fn: Any) -> Any:
    raise OperationalError(19, "constraint failed")


def _fake_connected_conn() -> DqliteConnection:
    conn = DqliteConnection("localhost:9001")
    conn._db_id = 1
    conn._protocol = object()  # type: ignore[assignment]
    return conn


@pytest.mark.asyncio
async def test_multi_statement_exec_partial_failure_sets_untracked_flag() -> None:
    """``BEGIN; SAVEPOINT a; INSERT (failing)`` → flag set."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("BEGIN; SAVEPOINT a; INSERT INTO t VALUES (?)", (1,))
    assert conn._has_untracked_savepoint is True


@pytest.mark.asyncio
async def test_multi_statement_no_tx_verbs_does_not_set_untracked_flag() -> None:
    """Negative pin: a benign multi-INSERT batch failing mid-batch
    does NOT trigger the flag (no tx-control verb anywhere)."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute(
            "INSERT INTO t VALUES (?); INSERT INTO t VALUES (?); INSERT INTO t VALUES (?)",
            (1,),
        )
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_single_statement_failure_does_not_set_flag() -> None:
    """Negative pin: a single-statement EXEC failure (no ``;``) skips
    the conservative-flag path entirely — the success-only tracker
    update is enough."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("INSERT INTO t VALUES (?)", (1,))
    assert conn._has_untracked_savepoint is False


@pytest.mark.asyncio
async def test_multi_statement_with_savepoint_only_sets_flag() -> None:
    """A multi-statement batch with SAVEPOINT but no other tx verb
    still sets the flag — SAVEPOINT alone can autobegin a tx."""
    conn = _fake_connected_conn()
    with (
        patch.object(conn, "_run_protocol", new=_raise_constraint),
        pytest.raises(OperationalError),
    ):
        await conn.execute("SAVEPOINT sp; INSERT INTO t VALUES (?)", (1,))
    assert conn._has_untracked_savepoint is True
