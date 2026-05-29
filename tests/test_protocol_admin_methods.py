"""Protocol-layer tests for ``DqliteProtocol.cluster`` and ``transfer``
(wire mocked; the request-encode + response-decode contract is the unit)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire import NodeRole
from dqlitewire.messages import (
    DbResponse,
    EmptyResponse,
    FailureResponse,
    LeaderResponse,
    ServersResponse,
)
from dqlitewire.messages.responses import NodeInfo


class TestProtocolCluster:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_cluster_returns_decoded_node_list(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """ServersResponse decodes into the list[NodeInfo] the method returns."""
        nodes = [
            NodeInfo(node_id=1, address="node1:9001", role=NodeRole.VOTER),
            NodeInfo(node_id=2, address="node2:9002", role=NodeRole.VOTER),
            NodeInfo(node_id=3, address="node3:9003", role=NodeRole.STANDBY),
        ]
        mock_reader.read.return_value = ServersResponse(nodes=nodes).encode()

        result = await protocol.cluster()

        assert result == nodes

    async def test_cluster_failure_response_raises_operational_error(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """A FailureResponse surfaces as OperationalError with code+message."""
        mock_reader.read.return_value = FailureResponse(
            code=1, message="cluster mid-shutdown"
        ).encode()

        with pytest.raises(OperationalError) as exc_info:
            await protocol.cluster()

        assert exc_info.value.code == 1
        assert "cluster mid-shutdown" in str(exc_info.value)

    async def test_cluster_wrong_response_type_raises_protocol_error(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """A wrong response type (wire drift) raises ProtocolError so the
        caller can invalidate the connection."""
        mock_reader.read.return_value = LeaderResponse(node_id=1, address="node1:9001").encode()

        with pytest.raises(ProtocolError, match="Expected ServersResponse"):
            await protocol.cluster()


class TestProtocolTransfer:
    @pytest.fixture
    def protocol(self, mock_reader: AsyncMock, mock_writer: MagicMock) -> DqliteProtocol:
        return DqliteProtocol(mock_reader, mock_writer)

    async def test_transfer_success_does_not_raise(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Healthy path: EmptyResponse decodes without raising (returns None)."""
        mock_reader.read.return_value = EmptyResponse().encode()

        await protocol.transfer(target_node_id=2)

    async def test_transfer_failure_response_raises_operational_error(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Server-side rejection surfaces as OperationalError."""
        mock_reader.read.return_value = FailureResponse(
            code=1, message="target not a voter"
        ).encode()

        with pytest.raises(OperationalError) as exc_info:
            await protocol.transfer(target_node_id=99)

        assert exc_info.value.code == 1
        assert "target not a voter" in str(exc_info.value)

    async def test_transfer_wrong_response_type_raises_protocol_error(
        self,
        protocol: DqliteProtocol,
        mock_reader: AsyncMock,
    ) -> None:
        """Wire-level drift surfaces as ProtocolError."""
        mock_reader.read.return_value = DbResponse(db_id=1).encode()

        with pytest.raises(ProtocolError, match="Expected EmptyResponse"):
            await protocol.transfer(target_node_id=2)
