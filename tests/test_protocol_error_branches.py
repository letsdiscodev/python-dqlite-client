"""Systematic coverage of each dispatcher method's two broken-server guards:
FailureResponse -> OperationalError (code+message preserved), and any other
wrong type -> ProtocolError citing the received type."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import (
    DbResponse,
    FailureResponse,
    LeaderResponse,
)


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer)


class TestHandshakeWrongResponseType:
    async def test_handshake_rejects_leader_response_instead_of_welcome(
        self, protocol: DqliteProtocol
    ) -> None:
        protocol._reader.read.return_value = LeaderResponse(  # type: ignore[attr-defined]
            node_id=1, address="a:1"
        ).encode()
        with pytest.raises(ProtocolError, match="Expected WelcomeResponse"):
            await protocol.handshake()


class TestGetLeaderErrorBranches:
    async def test_get_leader_failure_response(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
            code=1, message="probe failed"
        ).encode()
        with pytest.raises(OperationalError) as exc_info:
            await protocol.get_leader()
        assert exc_info.value.code == 1
        assert "probe failed" in exc_info.value.message

    async def test_get_leader_wrong_type(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = DbResponse(db_id=1).encode()  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError, match="Expected LeaderResponse"):
            await protocol.get_leader()


class TestOpenDatabaseWrongType:
    async def test_open_database_wrong_type(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = LeaderResponse(  # type: ignore[attr-defined]
            node_id=1, address="a:1"
        ).encode()
        with pytest.raises(ProtocolError, match="Expected DbResponse"):
            await protocol.open_database("test.db")


class TestPrepareErrorBranches:
    async def test_prepare_failure_response(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
            code=1, message="SQL syntax error"
        ).encode()
        with pytest.raises(OperationalError) as exc_info:
            await protocol.prepare(1, "BAD SQL")
        assert exc_info.value.code == 1

    async def test_prepare_wrong_type(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = DbResponse(db_id=1).encode()  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError, match="Expected StmtResponse"):
            await protocol.prepare(1, "SELECT 1")


class TestFinalizeFailureBranch:
    async def test_finalize_failure_response(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
            code=21, message="misuse"
        ).encode()
        with pytest.raises(OperationalError) as exc_info:
            await protocol.finalize(1, 1)
        assert exc_info.value.code == 21


class TestExecSqlErrorBranches:
    async def test_exec_sql_failure_response(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
            code=19, message="constraint failed"
        ).encode()
        with pytest.raises(OperationalError) as exc_info:
            await protocol.exec_sql(1, "INSERT INTO t VALUES (1)")
        assert exc_info.value.code == 19

    async def test_exec_sql_wrong_type(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = DbResponse(db_id=1).encode()  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError, match="Expected ResultResponse"):
            await protocol.exec_sql(1, "INSERT INTO t VALUES (1)")


class TestSendQueryErrorBranches:
    async def test_query_sql_failure_response(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = FailureResponse(  # type: ignore[attr-defined]
            code=5, message="busy"
        ).encode()
        with pytest.raises(OperationalError) as exc_info:
            await protocol.query_sql(1, "SELECT 1")
        assert exc_info.value.code == 5

    async def test_query_sql_wrong_type(self, protocol: DqliteProtocol) -> None:
        protocol._reader.read.return_value = DbResponse(db_id=1).encode()  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError, match="Expected RowsResponse"):
            await protocol.query_sql(1, "SELECT 1")
