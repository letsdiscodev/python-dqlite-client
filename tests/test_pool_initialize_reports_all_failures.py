"""``ConnectionPool.initialize`` logs every failure at WARNING; multiple
failures raise a ``BaseExceptionGroup``, a single one re-raises its narrow
type so existing ``except`` paths keep matching."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_initialize_single_failure_raises_narrow_type(monkeypatch) -> None:
    """One failure re-raises the narrow exception type."""
    pool = ConnectionPool(
        ["localhost:9001"],
        min_size=1,
        max_size=1,
        timeout=0.5,
    )

    async def _fail() -> Any:
        raise DqliteConnectionError("refused")

    monkeypatch.setattr(pool, "_create_connection", _fail)

    with pytest.raises(DqliteConnectionError, match="refused"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_initialize_multiple_failures_raises_exception_group(monkeypatch, caplog) -> None:
    """Three failures raise a ``BaseExceptionGroup``, each logged at WARNING."""
    pool = ConnectionPool(
        ["a:9001", "b:9001", "c:9001"],
        min_size=3,
        max_size=3,
        timeout=0.5,
    )

    failures_iter = iter(
        [
            TimeoutError("timeout"),
            DqliteConnectionError("refused"),
            ConnectionError("peer rst"),
        ]
    )

    async def _fail() -> Any:
        raise next(failures_iter)

    monkeypatch.setattr(pool, "_create_connection", _fail)

    caplog.set_level(logging.WARNING, logger="dqliteclient.pool")
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await pool.initialize()
    eg = exc_info.value
    assert len(eg.exceptions) == 3
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("create_connection" in m for m in warning_msgs) == 3
