"""``transaction()`` emits a bare ``BEGIN``: dqlite's Raft FSM serializes regardless of the
qualifier (always SERIALIZABLE), and a qualifier would diverge from the Go / C peer clients."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dqliteclient.connection import _TRANSACTION_BEGIN_SQL, DqliteConnection


@pytest.fixture
def conn() -> DqliteConnection:
    return DqliteConnection("localhost:9001")


def test_transaction_begin_sql_constant_is_bare_begin() -> None:
    assert _TRANSACTION_BEGIN_SQL == "BEGIN"


class TestTransactionEmitsBareBegin:
    async def test_transaction_emits_bare_begin(self, conn: DqliteConnection) -> None:
        execute = AsyncMock(return_value=(0, 0))
        conn.execute = execute

        async with conn.transaction():
            pass

        sqls = [call.args[0] for call in execute.call_args_list]
        assert sqls == ["BEGIN", "COMMIT"]

    async def test_no_immediate_or_exclusive_qualifier(self, conn: DqliteConnection) -> None:
        execute = AsyncMock(return_value=(0, 0))
        conn.execute = execute

        async with conn.transaction():
            pass

        for call in execute.call_args_list:
            sql: Any = call.args[0]
            assert sql.upper() != "BEGIN IMMEDIATE"
            assert sql.upper() != "BEGIN EXCLUSIVE"
            assert sql.upper() != "BEGIN DEFERRED"
