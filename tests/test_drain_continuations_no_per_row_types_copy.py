"""Pin: ``DqliteProtocol._drain_continuations`` accumulates
``row_types`` and ``rows`` via direct list-extend without the
per-row defensive ``list(rt)`` copy.

The wire layer constructs each ``RowsResponse`` fresh per decode
(``responses.RowsResponse._from_decoded`` builds a new instance
with fresh inner lists) and ``_drain_continuations`` discards the
instances inside its own scope — nothing else holds a reference,
so aliasing the inner lists into the cumulative accumulator is
safe.

The prior shape ran ``[list(rt) for rt in initial.row_types]``
per frame on the loop thread. For a 10k-row × 32-col frame this
was ~10k fresh list allocations + 10k × 32-element memcpys on the
loop thread between the wire decode and the next ``await
asyncio.sleep(0)`` — ~5-10 ms per frame of pure-Python allocator
pressure that an ownership-transfer extend avoids entirely.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient import protocol as protocol_mod


def _drain_continuations_source() -> str:
    return textwrap.dedent(inspect.getsource(protocol_mod.DqliteProtocol._drain_continuations))


def test_drain_continuations_does_not_copy_per_row_types_list() -> None:
    """The fix removes both ``[list(rt) for rt in initial.row_types]``
    and ``list(rt) for rt in next_response.row_types`` — the
    accumulator extends directly from the wire-layer lists.
    """
    src = _drain_continuations_source()
    tree = ast.parse(src)

    per_row_list_copies = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Look for ``list(rt)`` where ``rt`` is a Name (the loop var).
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "rt"
        ):
            per_row_list_copies += 1

    assert per_row_list_copies == 0, (
        f"_drain_continuations still contains {per_row_list_copies} "
        "per-row ``list(rt)`` copies; the accumulator should take "
        "ownership of the wire-layer lists via direct ``extend``."
    )


def test_drain_continuations_aliases_initial_row_types_into_accumulator() -> None:
    """Behavioural pin: after the fix, the accumulator's first
    row-types entries are the SAME identity as the initial frame's
    row-types entries (no defensive copy). This proves the
    ownership-transfer is in effect.

    We exercise the function via the wire-layer ``RowsResponse``
    decoder to construct realistic inputs without relying on
    protocol-level scaffolding.
    """
    import asyncio
    from unittest.mock import MagicMock

    from dqlitewire.constants import ValueType
    from dqlitewire.messages.responses import RowsResponse

    # Construct a synthetic RowsResponse via the wire-layer
    # ``_from_decoded`` constructor so the inner lists have the
    # same provenance as the production decode path.
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
    # Ownership transfer: the accumulator holds the SAME list
    # objects the wire layer produced.
    assert row_types[0] is initial.row_types[0], (
        "row_types[0] should be aliased from initial.row_types[0] after "
        "the ownership-transfer fix; got a defensive copy instead."
    )


def test_drain_continuations_passes_rows_correctly() -> None:
    """Regression guard: behaviour unchanged for the single-frame
    case (no continuation reads required).
    """
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
