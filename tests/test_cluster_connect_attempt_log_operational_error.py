"""Pin: ``try_connect``'s widened ``except`` tuple includes
``OperationalError`` so a non-leader-flip failure (e.g.,
``SQLITE_NOTFOUND`` from ``open_database``'s ``FailureResponse``
arm) emits the per-attempt DEBUG breadcrumb AND propagates
unwrapped (the retry classifier deliberately does NOT retry these
codes).

A regression that drops ``OperationalError`` from the catch tuple
would silently lose the breadcrumb; a regression that rewraps it
to ``DqliteConnectionError`` would silently widen the retry surface
to non-retryable failures.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.exceptions import OperationalError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_try_connect_operationalerror_emits_attempt_log_and_propagates_unwrapped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cluster = ClusterClient(
        MemoryNodeStore(["leader:9001"]),
        timeout=2.0,
    )
    cluster.find_leader = AsyncMock(return_value="leader:9001")

    async def _raise_operational_error(*args: object, **kwargs: object) -> object:
        # Non-leader-flip code: SQLITE_NOTFOUND (12) "no database opened".
        raise OperationalError("unknown database", code=12)

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    with (
        patch.object(DqliteConnection, "connect", AsyncMock(side_effect=_raise_operational_error)),
        pytest.raises(OperationalError) as exc_info,
    ):
        await cluster.connect(max_attempts=1)

    # Propagates unwrapped — NOT rewrapped to DqliteConnectionError.
    assert exc_info.value.code == 12
    assert "unknown database" in str(exc_info.value)

    # Per-attempt DEBUG breadcrumb emitted.
    debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert any("ClusterClient.connect attempt" in r.message for r in debug_records), (
        "Expected per-attempt DEBUG breadcrumb. Got log records: "
        f"{[r.message for r in caplog.records]}"
    )
