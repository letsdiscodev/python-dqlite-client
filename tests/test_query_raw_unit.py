"""Pin: ``DqliteConnection.query_raw`` and ``query_raw_typed``
validate-before-``_run_protocol`` ordering.

The earlier shape-pin tests in this file
(``test_query_raw_returns_two_tuple`` / ``..._typed_returns_four_tuple``)
monkeypatched ``_run_protocol`` with a fake that ignored its callable
argument and returned canned values, then asserted the wrapper
returned the same canned values. That pin was tautological — a
regression that swapped ``query_sql`` for ``query_sql_typed`` (or
vice versa) would not change the fake's return and the test would
still pass. The integration tests at ``tests/integration`` already
cover the actual return-shape contract end-to-end.

The remaining tests pin the validate-first ordering: a Mapping
passed as ``params`` raises ``DataError`` without touching the wire.
The fake ``_run_protocol`` is configured with
``side_effect=AssertionError`` so any protocol invocation surfaces
the bug; this is a real ordering pin (not a tautology).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import DataError


def _make_conn() -> DqliteConnection:
    """Skeleton DqliteConnection sufficient for the validate +
    _run_protocol-mock paths; bypasses actual transport."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._closed = False
    conn._db_id = 1
    conn._protocol = MagicMock()
    return conn


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
