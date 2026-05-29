"""Mid-stream FailureResponse must surface as OperationalError(message, code).

Wrapping it as ProtocolError flattens the SQLite code into a string, breaking
sqlalchemy-dqlite's is_disconnect check (needs OperationalError with .code).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import OperationalError, ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.constants import ROW_PART_MARKER, ValueType
from dqlitewire.messages import FailureResponse, RowsResponse
from dqlitewire.types import encode_uint64


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=1.0, address="test:9001")


class TestServerFailureMidStreamClassification:
    async def test_mid_stream_failure_raises_operational_error_with_code(
        self, protocol: DqliteProtocol
    ) -> None:
        # Initial frame: row with PART marker signaling more to come.
        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        part_marker = encode_uint64(ROW_PART_MARKER)
        frame_with_part = initial[:-8] + part_marker

        # SQLITE_IOERR_NOT_LEADER = 10250.
        failure = FailureResponse(code=10250, message="not leader")
        failure_frame = failure.encode()

        protocol._reader.read = AsyncMock(side_effect=[frame_with_part + failure_frame, b""])

        with pytest.raises(OperationalError) as exc_info:
            await protocol.query_sql(1, "SELECT 1")

        assert exc_info.value.code == 10250
        assert "not leader" in exc_info.value.message

    async def test_mid_stream_non_leader_failure_keeps_connection_usable(
        self, protocol: DqliteProtocol
    ) -> None:
        """After a mid-stream non-leader FailureResponse the buffer is not
        poisoned, so the next request on the same connection decodes cleanly."""
        from dqlitewire.messages import LeaderResponse

        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        part_marker = encode_uint64(ROW_PART_MARKER)
        frame_with_part = initial[:-8] + part_marker

        # SQLITE_CONSTRAINT = 19.
        failure_frame = FailureResponse(code=19, message="CHECK constraint failed").encode()

        followup_frame = LeaderResponse(node_id=1, address="127.0.0.1:9001").encode()

        protocol._reader.read = AsyncMock(
            side_effect=[frame_with_part + failure_frame, followup_frame, b""]
        )

        with pytest.raises(OperationalError) as exc_info:
            await protocol.query_sql(1, "SELECT 1")
        assert exc_info.value.code == 19

        assert not protocol._decoder.is_poisoned
        leader = await protocol._read_response()
        assert isinstance(leader, LeaderResponse)
        assert leader.address == "127.0.0.1:9001"

    async def test_mid_stream_failure_not_raised_as_protocol_error(
        self, protocol: DqliteProtocol
    ) -> None:
        """Regression guard: the raised exception must not be ProtocolError
        (except via OperationalError's own MRO)."""
        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,
        ).encode()
        part_marker = encode_uint64(ROW_PART_MARKER)
        frame_with_part = initial[:-8] + part_marker

        failure_frame = FailureResponse(code=5, message="db locked").encode()
        protocol._reader.read = AsyncMock(side_effect=[frame_with_part + failure_frame, b""])

        with pytest.raises(OperationalError) as exc_info:
            await protocol.query_sql(1, "SELECT 1")

        assert not isinstance(exc_info.value, ProtocolError)
