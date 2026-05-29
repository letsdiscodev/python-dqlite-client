"""Pin: ``_read_continuation`` wraps ``_WireProtocolError`` into the client-layer
``ProtocolError`` so multi-frame callers' ``except ProtocolError`` boundaries hold."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.constants import ROW_PART_MARKER
from dqlitewire.messages import RowsResponse


@pytest.fixture
def protocol() -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer, timeout=1.0)


class TestReadContinuationWrapsWireError:
    async def test_malformed_continuation_bytes_raise_client_protocol_error(
        self, protocol: DqliteProtocol
    ) -> None:
        """Garbage continuation bytes after a PART marker must surface as the
        client-layer ``ProtocolError``."""
        import struct

        from dqlitewire.constants import ValueType
        from dqlitewire.types import encode_uint64

        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,  # encoded below manually
        ).encode()
        # Swap the trailing DONE marker to PART so the decoder expects a continuation frame.
        part_marker = encode_uint64(ROW_PART_MARKER)
        done_marker_prefix = initial[:-8]
        frame_with_part = done_marker_prefix + part_marker

        # Continuation header: size_words(u32), msg_type(u8), schema(u8), reserved(u16).
        garbage_header = struct.pack("<IBBH", 1, 0, 0, 0)
        garbage_body = b"\x00" * 8
        garbage_frame = garbage_header + garbage_body

        protocol._reader.read = AsyncMock(side_effect=[frame_with_part + garbage_frame, b""])

        with pytest.raises(ProtocolError):
            await protocol.query_sql(1, "SELECT 1")
