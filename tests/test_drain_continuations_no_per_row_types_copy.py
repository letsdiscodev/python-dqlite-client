"""_drain_continuations extends rows/row_types directly without a per-row ``list(rt)`` copy.

The wire layer builds each RowsResponse fresh and the drain discards the instances in its own
scope, so aliasing the inner lists into the accumulator is safe and avoids per-frame allocator
pressure (~10k list allocs + memcpys for a 10k-row x 32-col frame on the loop thread).
"""

from __future__ import annotations

from dqliteclient import protocol as protocol_mod


def test_drain_continuations_aliases_initial_row_types_into_accumulator() -> None:
    """Accumulator row-types entries are the SAME identity as the initial frame's (no copy)."""
    import asyncio
    from unittest.mock import MagicMock

    from dqlitewire.constants import ValueType
    from dqlitewire.messages.responses import RowsResponse

    # Build via ``_from_decoded`` so the inner lists share the production decode provenance.
    initial = RowsResponse._from_decoded(
        column_names=["a", "b"],
        column_types=[ValueType.INTEGER, ValueType.TEXT],
        rows=[[1, "hello"]],
        row_types=[[ValueType.INTEGER, ValueType.TEXT]],
        has_more=False,
    )

    proto = protocol_mod.DqliteProtocol.__new__(protocol_mod.DqliteProtocol)
    proto._max_total_rows = None
    proto._max_continuation_frames = None
    proto._reader = MagicMock()
    proto._writer = MagicMock()

    rows, row_types = asyncio.run(proto._drain_continuations(initial, deadline=0.0))
    assert rows == [[1, "hello"]]
    assert row_types == [[ValueType.INTEGER, ValueType.TEXT]]
    assert row_types[0] is initial.row_types[0], (
        "row_types[0] should be aliased from initial.row_types[0] after "
        "the ownership-transfer fix; got a defensive copy instead."
    )


def test_drain_continuations_passes_rows_correctly() -> None:
    import asyncio
    from unittest.mock import MagicMock

    from dqlitewire.constants import ValueType
    from dqlitewire.messages.responses import RowsResponse

    initial = RowsResponse._from_decoded(
        column_names=["x"],
        column_types=[ValueType.INTEGER],
        rows=[[1], [2], [3]],
        row_types=[
            [ValueType.INTEGER],
            [ValueType.INTEGER],
            [ValueType.INTEGER],
        ],
        has_more=False,
    )

    proto = protocol_mod.DqliteProtocol.__new__(protocol_mod.DqliteProtocol)
    proto._max_total_rows = None
    proto._max_continuation_frames = None
    proto._reader = MagicMock()
    proto._writer = MagicMock()

    rows, row_types = asyncio.run(proto._drain_continuations(initial, deadline=0.0))
    assert rows == [[1], [2], [3]]
    assert len(row_types) == 3
