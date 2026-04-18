"""max_total_rows cap actually fires in the continuation-drain loop.

Cycle 9 wired the cap through the layers; cycle 16 adds the missing
test that exercises the enforcement itself. Uses a mocked protocol
response stream so we don't need a cluster to deliver millions of rows.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import ProtocolError
from dqliteclient.protocol import DqliteProtocol


def _make_rows_response(rows: list[list[object]], has_more: bool) -> MagicMock:
    r = MagicMock(name="RowsResponse")
    r.rows = rows
    r.has_more = has_more
    r.column_names = ["v"]
    r.column_types = [1]  # INTEGER
    return r


class TestMaxTotalRowsEnforcement:
    def test_exceeding_cap_raises_protocol_error(self) -> None:
        """A continuation frame that pushes us past max_total_rows raises."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=5)

        initial = _make_rows_response([[1], [2], [3]], has_more=True)
        # This next frame would bring total to 6, exceeding the cap of 5.
        over_cap = _make_rows_response([[4], [5], [6]], has_more=False)

        p._read_continuation = AsyncMock(return_value=over_cap)  # type: ignore[method-assign]

        async def run() -> None:
            await p._drain_continuations(initial, deadline=999999.0)

        with pytest.raises(ProtocolError, match="max_total_rows"):
            asyncio.run(run())

    def test_exactly_at_cap_does_not_raise(self) -> None:
        """Hitting the cap exactly is fine; only exceeding raises."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=5)

        initial = _make_rows_response([[1], [2], [3]], has_more=True)
        at_cap = _make_rows_response([[4], [5]], has_more=False)  # total: 5

        p._read_continuation = AsyncMock(return_value=at_cap)  # type: ignore[method-assign]

        rows = asyncio.run(p._drain_continuations(initial, deadline=999999.0))
        assert len(rows) == 5

    def test_none_disables_cap(self) -> None:
        """max_total_rows=None means the cap never fires."""
        reader = MagicMock()
        writer = MagicMock()
        p = DqliteProtocol(reader, writer, timeout=5.0, max_total_rows=None)

        initial = _make_rows_response([[i] for i in range(100)], has_more=True)
        big = _make_rows_response([[i] for i in range(100, 10_000)], has_more=False)

        p._read_continuation = AsyncMock(return_value=big)  # type: ignore[method-assign]

        rows = asyncio.run(p._drain_continuations(initial, deadline=999999.0))
        assert len(rows) == 10_000
