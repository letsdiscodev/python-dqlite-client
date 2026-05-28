"""Pin: ``DqliteConnection.fetch`` yields cooperatively while building
its list-of-dicts result, so a large query result does not monopolise
the event loop after the (already loop-friendly) wire decode completes.

``fetch`` reshapes the decoded ``rows`` into ``[dict(zip(columns, row))
for row in rows]``. The prior single comprehension ran entirely on the
loop thread with no suspension point; the result-set size is bounded
only by ``max_total_rows`` (default ``DEFAULT_MAX_TOTAL_ROWS =
10_000_000``), so a multi-100k/multi-million-row result froze the loop
for the whole reshape — starving heartbeats, pool acquirers, and any
concurrent connection sharing the loop, and preventing cancellation from
landing.

The decode side (``_read_response`` / ``_read_continuation`` /
``_drain_continuations``) already yields per frame, but those run inside
``_run_protocol``; the post-decode reshape in ``fetch`` is downstream of
that and was the residual unguarded per-row Python pass. ``fetchall``
(returns the rows directly) and ``fetchval`` (indexes ``rows[0][0]``) do
no per-row build and are unaffected; ``fetchone`` inherits the yield via
``fetch``.

The fix gates the yield on ``len(rows) >= _FETCH_DICT_YIELD_EVERY`` —
below it the straight comprehension runs with zero scheduler overhead —
and fires ``await asyncio.sleep(0)`` every ``_FETCH_DICT_YIELD_EVERY``
rows. The ``strict=True`` zip semantics (arity mismatch raises
``ValueError``) and the dict contents/order are preserved byte-identical.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dqliteclient.connection import DqliteConnection


def _make_conn_returning(columns: list[str], rows: list[list[Any]]) -> DqliteConnection:
    conn = DqliteConnection("localhost:9001", timeout=2.0)

    async def fake_run_protocol(_op: Any, *a: Any, **k: Any) -> tuple[list[str], list[list[Any]]]:
        return columns, rows

    conn._run_protocol = fake_run_protocol  # type: ignore[assignment]
    return conn


@pytest.mark.asyncio
async def test_fetch_large_result_yields_between_batches() -> None:
    """A 50k-row fetch must let a sibling ticker run a non-trivial number
    of times. Under the prior single comprehension the sibling got zero
    ticks."""
    columns = ["a", "b"]
    rows = [[i, i * 2] for i in range(50_000)]
    conn = _make_conn_returning(columns, rows)

    sibling_ran = 0

    async def sibling() -> None:
        nonlocal sibling_ran
        while True:
            await asyncio.sleep(0)
            sibling_ran += 1

    task = asyncio.create_task(sibling())
    try:
        await asyncio.sleep(0)
        baseline = sibling_ran
        result = await conn.fetch("SELECT a, b FROM t")
        during = sibling_ran - baseline
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(result) == 50_000
    assert result[0] == {"a": 0, "b": 0}
    assert result[-1] == {"a": 49_999, "b": 99_998}
    assert during >= 5, (
        f"sibling ran only {during} times during a 50k-row fetch; "
        "cooperative yield missing on the dict-materialization path"
    )


@pytest.mark.asyncio
async def test_fetch_small_result_no_yield() -> None:
    """Small results (below the threshold) must not pay any yield
    overhead — keep the straight comprehension."""
    columns = ["a"]
    rows = [[i] for i in range(100)]
    conn = _make_conn_returning(columns, rows)

    sibling_ran = 0

    async def sibling() -> None:
        nonlocal sibling_ran
        while True:
            await asyncio.sleep(0)
            sibling_ran += 1

    task = asyncio.create_task(sibling())
    try:
        await asyncio.sleep(0)
        baseline = sibling_ran
        result = await conn.fetch("SELECT a FROM t")
        during = sibling_ran - baseline
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(result) == 100
    assert during <= 2, f"small fetch should not yield internally; sibling ran {during} times"


@pytest.mark.asyncio
async def test_fetch_large_result_dict_identity() -> None:
    """The yielding build must be byte-identical to the prior
    comprehension (same dicts, same order)."""
    columns = ["x", "y"]
    rows = [[i, str(i)] for i in range(10_000)]
    conn = _make_conn_returning(columns, rows)

    result = await conn.fetch("SELECT x, y FROM t")

    assert result == [{"x": i, "y": str(i)} for i in range(10_000)]


@pytest.mark.asyncio
async def test_fetch_large_result_preserves_strict_zip() -> None:
    """``strict=True`` must still raise on a column/row arity mismatch,
    even on the large (yielding) path."""
    columns = ["a", "b"]
    rows: list[list[Any]] = [[i, i] for i in range(8_000)]
    rows[5_000] = [42]  # arity mismatch, past the first yield boundary
    conn = _make_conn_returning(columns, rows)

    with pytest.raises(ValueError):
        await conn.fetch("SELECT a, b FROM t")


def test_fetch_dict_yield_constant_is_defined_locally() -> None:
    """The yield stride must be a local client constant (no dbapi import,
    which would invert the dependency direction)."""
    from dqliteclient import connection as conn_mod

    assert isinstance(conn_mod._FETCH_DICT_YIELD_EVERY, int)
    assert conn_mod._FETCH_DICT_YIELD_EVERY > 0
