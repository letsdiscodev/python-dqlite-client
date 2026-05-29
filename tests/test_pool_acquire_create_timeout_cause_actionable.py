"""Pin: the fresh-slot create-connection clamp in ``acquire`` chains its
``DqliteConnectionError`` from a meaningful ``TimeoutError`` (the original
from ``asyncio.timeout`` or a synthesised one with actionable text), so
``exc.__cause__`` carries a forensic signal for pool-saturation triage.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.exceptions import DqliteConnectionError
from dqliteclient.pool import ConnectionPool


@pytest.mark.asyncio
async def test_fresh_slot_create_clamp_translates_at_clamp_scope() -> None:
    """A fresh-slot reservation with a slow ``_create_connection`` must
    surface as ``DqliteConnectionError("Timed out ...") from TimeoutError``."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.05)

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied")

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        patch.object(pool, "_drain_idle", new=AsyncMock()),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with pool.acquire():
            pytest.fail("should not reach")

    err = exc_info.value
    assert "Timed out creating a fresh connection from the pool" in str(err)
    assert isinstance(err.__cause__, TimeoutError), (
        f"expected DqliteConnectionError chained from TimeoutError; got __cause__={err.__cause__!r}"
    )


@pytest.mark.asyncio
async def test_already_expired_deadline_carries_actionable_cause_text() -> None:
    """When the deadline is already past before the clamp scope opens, the
    ``DqliteConnectionError`` is chained from a TimeoutError naming the
    overshoot. Either branch may land given timing; the load-bearing pin is
    that a TimeoutError is present in the chain."""
    pool = ConnectionPool(["localhost:9001"], max_size=1, timeout=0.001)

    async def _slow_create() -> Any:
        await asyncio.sleep(2.0)
        raise AssertionError("clamp not applied")

    with (
        patch.object(pool, "_create_connection", new=_slow_create),
        patch.object(pool, "_drain_idle", new=AsyncMock()),
        pytest.raises(DqliteConnectionError) as exc_info,
    ):
        async with pool.acquire():
            pytest.fail("should not reach")

    err = exc_info.value
    assert isinstance(err.__cause__, TimeoutError)
