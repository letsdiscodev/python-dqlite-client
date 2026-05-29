"""An outer cancel landing before connect's first attempt propagates cleanly with
no breadcrumb. Trips if a CancelledError catch in try_connect synthesises one,
violating structured-concurrency cancel propagation."""

from __future__ import annotations

import asyncio
import logging

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_outer_cancel_before_first_attempt_does_not_log_breadcrumb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """timeout(0) lands the cancel before any try_connect; no breadcrumb fires."""
    cluster = ClusterClient(MemoryNodeStore(["10.0.0.1:9001"]))

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    with pytest.raises((TimeoutError, asyncio.CancelledError)):
        async with asyncio.timeout(0):
            await cluster.connect()

    # Pin on the specific breadcrumb phrases; bare "attempt" is too broad.
    breadcrumb_phrases = ("attempt failed", "Connection attempt")
    matching = [
        r for r in caplog.records if any(phrase in r.getMessage() for phrase in breadcrumb_phrases)
    ]
    assert not matching, (
        "Outer cancel-before-first-attempt produced per-attempt log records "
        f"({len(matching)}); a defensive CancelledError catch in try_connect "
        "is suppressing the structured-concurrency cancel contract. "
        "Records: " + " | ".join(r.getMessage() for r in matching)
    )
