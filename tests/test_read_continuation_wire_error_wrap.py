"""Regression test for the ``_read_continuation`` error wrap.

The initial-response decode path wraps ``_WireProtocolError`` into the
client-layer ``ProtocolError`` (tested elsewhere). The continuation-
frame path (``_read_continuation``) has the same wrap at protocol.py
lines 383-384 but was not covered by a dedicated test. A regression
would leak the wire exception out of the client layer, breaking
``except ProtocolError`` boundaries for callers that assemble
multi-frame result sets.
"""

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
        """Feed a valid initial ``RowsResponse(has_more=True)`` followed
        by garbage continuation bytes; the resulting wire-level
        ``ProtocolError`` must be wrapped into the client layer's
        ``ProtocolError`` so ``except dqliteclient.ProtocolError``
        continues to work.
        """
        import struct

        from dqlitewire.constants import ValueType
        from dqlitewire.types import encode_uint64

        # Initial frame: a 1-column RowsResponse with one row, followed
        # by a PART marker to signal more rows in the next frame.
        initial = RowsResponse(
            column_names=["x"],
            column_types=[ValueType.INTEGER],
            rows=[[1]],
            row_types=[[ValueType.INTEGER]],
            has_more=False,  # encoded below manually
        ).encode()
        # Swap the trailing DONE marker to PART so the decoder expects
        # a continuation frame.
        part_marker = encode_uint64(ROW_PART_MARKER)
        done_marker_prefix = initial[:-8]
        frame_with_part = done_marker_prefix + part_marker

        # A continuation frame starts with a header: size_words (uint32),
        # msg_type (uint8), schema (uint8), reserved (uint16), then body.
        # Feed 40 bytes of random garbage as a "continuation frame".
        garbage_header = struct.pack("<IBBH", 1, 0, 0, 0)
        garbage_body = b"\x00" * 8
        garbage_frame = garbage_header + garbage_body

        # Reader returns the valid initial then the garbage in one shot.
        protocol._reader.read = AsyncMock(side_effect=[frame_with_part + garbage_frame, b""])

        # Kick off a query; any read past the PART marker should trip
        # the wire decoder and surface as client ProtocolError via
        # _read_continuation.
        with pytest.raises(ProtocolError):
            await protocol.query_sql(1, "SELECT 1")
