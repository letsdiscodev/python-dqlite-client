"""Pin the DEBUG-log breadcrumb on ConnectionPool.initialize partial failure.

When ``_create_connection`` raises for one slot mid-gather, we close
the connections that did succeed and re-raise. Without a DEBUG line
the "we closed N survivors" outcome is invisible to operators — they
see only the re-raised failure, not the count or composition of the
silent close loop.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_partial_failure_logs_survivor_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = ConnectionPool(["localhost:19001"], min_size=3, max_size=3, timeout=0.5)

    # Build 2 succeeding connection mocks and 1 failing call so the
    # failing branch runs. Each succeeding mock records its close()
    # call so the test can assert the close loop ran.
    good1 = MagicMock()
    good1.close = AsyncMock()
    good2 = MagicMock()
    good2.close = AsyncMock()

    def _seq_side_effect() -> AsyncMock:
        calls = [good1, good2, None]
        exc = DqliteConnectionError("peer reset")

        async def _create() -> object:
            slot = calls.pop(0)
            if slot is None:
                raise exc
            return slot

        return AsyncMock(side_effect=_create())

    call_results = [good1, good2, DqliteConnectionError("peer reset")]

    async def _create_mock() -> object:
        val = call_results.pop(0)
        if isinstance(val, BaseException):
            raise val
        return val

    pool._create_connection = _create_mock  # type: ignore[method-assign]

    with (
        caplog.at_level(logging.DEBUG, logger="dqliteclient.pool"),
        pytest.raises(DqliteConnectionError, match="peer reset"),
    ):
        await pool.initialize()

    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "aborting after 2/3" in r.getMessage()
    ]
    assert matching, f"expected DEBUG 'aborting after 2/3'; got {caplog.records!r}"

    # Survivors must have been close()'d.
    good1.close.assert_awaited_once()
    good2.close.assert_awaited_once()
