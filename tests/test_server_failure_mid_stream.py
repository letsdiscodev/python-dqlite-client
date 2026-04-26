"""Mid-stream FailureResponse must surface as client OperationalError.

The wire layer raises ``dqlitewire.ServerFailure(code, message)`` when a
FAILURE response arrives during a ROWS continuation stream.
``ServerFailure`` inherits from ``dqlitewire.ProtocolError``; without a
dedicated catch, the continuation-decode path at ``protocol.py`` wraps
it as ``client.ProtocolError("Wire decode failed: [code] message")``
and the SQLite error code is flattened into a string.

The dbapi layer then maps ``client.ProtocolError`` to
``InterfaceError``, breaking ``sqlalchemy-dqlite``'s ``is_disconnect``
check which tests ``isinstance(e, OperationalError) and e.code in
_LEADER_CHANGE_CODES``. Leadership-loss during a streaming SELECT would
not trigger SA pool invalidation.

Catching ``ServerFailure`` before the generic ``_WireProtocolError``
branch and re-raising as ``client.OperationalError(code, message)``
preserves the code for downstream classification.
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
        """FAILURE arriving during ROWS continuation must surface as
        ``OperationalError(code, message)`` — not ``ProtocolError`` —
        so the SQLite code reaches ``is_disconnect`` classifiers.
        """
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

        # Continuation frame: a FailureResponse (server errored mid-stream).
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
        """After a non-leader-class FailureResponse arriving mid-stream
        (e.g. SQLITE_CONSTRAINT, SQLITE_BUSY engine-side, SQLITE_ERROR),
        the wire decoder no longer poisons the buffer (see the wire-layer
        fix). The next request on the same connection must decode
        cleanly without ``ProtocolError("Wire decode failed: buffer
        poisoned")``."""
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

        # Mid-stream non-leader failure (SQLITE_CONSTRAINT = 19).
        failure_frame = FailureResponse(code=19, message="CHECK constraint failed").encode()

        # Follow-up response to the next request on the same connection.
        followup_frame = LeaderResponse(node_id=1, address="127.0.0.1:9001").encode()

        protocol._reader.read = AsyncMock(
            side_effect=[frame_with_part + failure_frame, followup_frame, b""]
        )

        with pytest.raises(OperationalError) as exc_info:
            await protocol.query_sql(1, "SELECT 1")
        assert exc_info.value.code == 19

        # Buffer is NOT poisoned; a second protocol operation succeeds.
        assert not protocol._decoder.is_poisoned
        leader = await protocol._read_response()
        assert isinstance(leader, LeaderResponse)
        assert leader.address == "127.0.0.1:9001"

    async def test_mid_stream_failure_not_raised_as_protocol_error(
        self, protocol: DqliteProtocol
    ) -> None:
        """The old behaviour was to wrap ServerFailure in ProtocolError.
        Guard against that regression: the raised exception must not be
        ``ProtocolError`` except via ``OperationalError``'s own MRO.
        """
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

        # OperationalError in this stack does not subclass ProtocolError.
        assert not isinstance(exc_info.value, ProtocolError)
