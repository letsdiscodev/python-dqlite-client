"""Pin: ``DqliteConnection.query_raw`` and ``query_raw_typed``
public API contracts — return shapes (2-tuple vs 4-tuple) and
the validate-before-_run_protocol ordering.

The methods route every dbapi cursor SELECT (via
``cursor.py:_call_client``) but had no direct unit tests; the
two source lines (one validate + one ``_run_protocol``-await
per method) were uncovered by the unit suite (only integration
tests exercised them).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError


def _make_conn() -> DqliteConnection:
    """Skeleton DqliteConnection sufficient for the validate +
    _run_protocol-mock paths; bypasses actual transport."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._closed = False  # type: ignore[attr-defined]
    conn._db_id = 1
    conn._protocol = MagicMock()
    return conn


@pytest.mark.asyncio
async def test_query_raw_returns_two_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    """``query_raw`` returns ``(column_names, rows)`` —
    ``(list[str], list[list[Any]])``."""
    conn = _make_conn()
    expected: tuple[list[str], list[list[Any]]] = (["c0", "c1"], [[1, "a"], [2, "b"]])

    async def _fake_run_protocol(_op: Any) -> tuple[list[str], list[list[Any]]]:
        return expected

    monkeypatch.setattr(conn, "_run_protocol", _fake_run_protocol)
    result = await conn.query_raw("SELECT 1, 'a'")
    assert result == expected
    assert len(result) == 2


@pytest.mark.asyncio
async def test_query_raw_typed_returns_four_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    """``query_raw_typed`` returns
    ``(column_names, column_types, row_types, rows)``."""
    conn = _make_conn()
    expected: tuple[list[str], list[int], list[list[int]], list[list[Any]]] = (
        ["c0"],
        [1],
        [[1], [1]],
        [[1], [2]],
    )

    async def _fake_run_protocol(
        _op: Any,
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        return expected

    monkeypatch.setattr(conn, "_run_protocol", _fake_run_protocol)
    result = await conn.query_raw_typed("SELECT 1")
    assert result == expected
    assert len(result) == 4


@pytest.mark.asyncio
async def test_query_raw_validates_params_before_run_protocol() -> None:
    """``_validate_params`` must run before ``_run_protocol``: a
    Mapping passed as ``params`` raises ``DataError`` without
    touching the wire — proves the validate-first ordering."""
    conn = _make_conn()
    fake_run = AsyncMock(side_effect=AssertionError("must not be called"))
    conn._run_protocol = fake_run

    with pytest.raises(DataError):
        await conn.query_raw("SELECT 1", {"a": 1})  # type: ignore[arg-type]
    fake_run.assert_not_called()


@pytest.mark.asyncio
async def test_query_raw_typed_validates_params_before_run_protocol() -> None:
    conn = _make_conn()
    fake_run = AsyncMock(side_effect=AssertionError("must not be called"))
    conn._run_protocol = fake_run

    with pytest.raises(DataError):
        await conn.query_raw_typed("SELECT 1", {1, 2})  # type: ignore[arg-type]
    fake_run.assert_not_called()
