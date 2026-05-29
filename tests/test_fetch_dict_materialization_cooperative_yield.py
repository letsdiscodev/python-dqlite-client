"""``DqliteConnection.fetch`` yields cooperatively while building its
list-of-dicts result, so a large result does not monopolise the loop."""

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
    of times."""
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
    overhead."""
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
    """The yielding build must produce the same dicts in the same order."""
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
    rows[5_000] = [42]  # arity mismatch past the first yield boundary
    conn = _make_conn_returning(columns, rows)

    with pytest.raises(ValueError):
        await conn.fetch("SELECT a, b FROM t")


def test_fetch_dict_yield_constant_is_defined_locally() -> None:
    """The yield stride must be a local client constant; a dbapi import
    would invert the dependency direction."""
    from dqliteclient import connection as conn_mod

    assert isinstance(conn_mod._FETCH_DICT_YIELD_EVERY, int)
    assert conn_mod._FETCH_DICT_YIELD_EVERY > 0
