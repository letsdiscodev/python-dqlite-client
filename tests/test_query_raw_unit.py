"""Pin: ``query_raw`` / ``query_raw_typed`` validate params before ``_run_protocol``.
``_run_protocol`` is stubbed with ``side_effect=AssertionError`` so any wire touch
surfaces the bug; a bad-``params`` call must raise DataError first."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError


def _make_conn() -> DqliteConnection:
    """Skeleton DqliteConnection for the validate + _run_protocol-mock paths."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._closed = False
    conn._db_id = 1
    conn._protocol = MagicMock()
    return conn


@pytest.mark.asyncio
async def test_query_raw_validates_params_before_run_protocol() -> None:
    """A Mapping passed as ``params`` raises DataError before any wire touch."""
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
