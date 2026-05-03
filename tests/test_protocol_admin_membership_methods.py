"""Protocol-layer tests for the new admin methods on ``DqliteProtocol``.

Covers ``add``, ``assign``, ``remove``, ``describe``, ``weight``,
``dump`` — the methods added to mirror go-dqlite's ``Client.Add`` /
``Assign`` / ``Remove`` / ``Describe`` / ``Weight`` / ``Dump``.

Sister of ``test_protocol_admin_methods.py`` (which covers the
earlier-added ``cluster`` and ``transfer``). The wire is mocked at
the ``mock_reader`` / ``mock_writer`` boundary; each test pins one
of: happy path, ``FailureResponse`` translation, wrong-response-
type detection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire import NodeRole
from dqlitewire.messages import (
    DbResponse,
    EmptyResponse,
    FailureResponse,
    FilesResponse,
    LeaderResponse,
    MetadataResponse,
)

# --- add ---


class TestProtocolAdd:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_add_success_returns_none(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = EmptyResponse().encode()
        await protocol.add(node_id=42, address="node42:9001")

    async def test_add_failure_response_raises_operational_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = FailureResponse(
            code=1, message="node already in cluster"
        ).encode()
        with pytest.raises(OperationalError) as excinfo:
            await protocol.add(node_id=42, address="node42:9001")
        assert excinfo.value.code == 1
        assert "already in cluster" in str(excinfo.value)

    async def test_add_wrong_response_type_raises_protocol_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = LeaderResponse(node_id=1, address="x:1").encode()
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.add(node_id=42, address="node42:9001")


# --- assign ---


class TestProtocolAssign:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_assign_success_returns_none(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = EmptyResponse().encode()
        await protocol.assign(node_id=42, role=NodeRole.VOTER)

    async def test_assign_failure_response_raises_operational_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = FailureResponse(code=1, message="node not found").encode()
        with pytest.raises(OperationalError):
            await protocol.assign(node_id=99, role=NodeRole.STANDBY)

    async def test_assign_wrong_response_type_raises_protocol_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = DbResponse(db_id=1).encode()
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.assign(node_id=42, role=NodeRole.VOTER)


# --- remove ---


class TestProtocolRemove:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_remove_success_returns_none(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = EmptyResponse().encode()
        await protocol.remove(node_id=42)

    async def test_remove_failure_response_raises_operational_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = FailureResponse(
            code=1, message="cannot remove leader"
        ).encode()
        with pytest.raises(OperationalError, match="cannot remove leader"):
            await protocol.remove(node_id=1)

    async def test_remove_wrong_response_type_raises_protocol_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = LeaderResponse(node_id=1, address="x:1").encode()
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.remove(node_id=42)


# --- describe ---


class TestProtocolDescribe:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_describe_returns_metadata_response(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = MetadataResponse(failure_domain=42, weight=7).encode()
        result = await protocol.describe()
        assert isinstance(result, MetadataResponse)
        assert result.failure_domain == 42
        assert result.weight == 7

    async def test_describe_failure_response_raises_operational_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = FailureResponse(code=1, message="busy").encode()
        with pytest.raises(OperationalError):
            await protocol.describe()

    async def test_describe_wrong_response_type_raises_protocol_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = EmptyResponse().encode()
        with pytest.raises(ProtocolError, match="Expected MetadataResponse"):
            await protocol.describe()


# --- weight ---


class TestProtocolWeight:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_weight_success_returns_none(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = EmptyResponse().encode()
        await protocol.weight(weight=5)

    async def test_weight_failure_response_raises_operational_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = FailureResponse(code=1, message="rejected").encode()
        with pytest.raises(OperationalError):
            await protocol.weight(weight=5)

    async def test_weight_wrong_response_type_raises_protocol_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = LeaderResponse(node_id=1, address="x:1").encode()
        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.weight(weight=5)


# --- dump ---


class TestProtocolDump:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_dump_returns_files_dict(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        # File contents must be 8-byte aligned per the wire-layer
        # invariant (mirrors upstream gateway.c::dumpFile).
        files = {"main": b"x" * 4096, "main-wal": b"y" * 8}
        mock_reader.read.return_value = FilesResponse(files=files).encode()
        result = await protocol.dump(database="main")
        assert result == files

    async def test_dump_failure_response_raises_operational_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = FailureResponse(code=1, message="unknown database").encode()
        with pytest.raises(OperationalError, match="unknown database"):
            await protocol.dump(database="missing")

    async def test_dump_wrong_response_type_raises_protocol_error(
        self, protocol: DqliteProtocol, mock_reader: AsyncMock
    ) -> None:
        mock_reader.read.return_value = EmptyResponse().encode()
        with pytest.raises(ProtocolError, match="Expected FilesResponse"):
            await protocol.dump(database="main")
