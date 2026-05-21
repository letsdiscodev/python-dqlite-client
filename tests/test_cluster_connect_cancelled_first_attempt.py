"""Pin: an outer cancel landing BEFORE
``ClusterClient.connect``'s first attempt runs propagates through
asyncio cleanly — no per-attempt log breadcrumb fires (because the
inner ``try_connect`` never ran), the cancel propagates as the
canonical Python idiom.

This is the structural divergence from go-dqlite's
``Connector.Connect`` which checks ``ctx.Done()`` only at
``attempt > 1`` and lets the FIRST attempt always run. Python's
asyncio cancel propagates through every ``await`` checkpoint —
the CORRECT shape for the Python ecosystem.

The pin trips if a future change adds a CancelledError catch in
``try_connect`` to synthesise a breadcrumb (which would violate
structured-concurrency cancel propagation).
"""

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
    """``asyncio.timeout(0)`` wrapping ``cluster.connect()`` lands
    the cancel BEFORE any ``try_connect`` invocation. The
    per-attempt DEBUG breadcrumb at ``cluster.py`` must NOT fire,
    confirming Python's structured-concurrency cancel propagation
    is preserved (no defensive CancelledError catch synthesises
    a "we tried once" breadcrumb)."""
    cluster = ClusterClient(MemoryNodeStore(["10.0.0.1:9001"]))

    caplog.set_level(logging.DEBUG, logger="dqliteclient.cluster")

    with pytest.raises((TimeoutError, asyncio.CancelledError)):
        async with asyncio.timeout(0):
            await cluster.connect()

    # The per-attempt log emitter at try_connect lives inside the
    # attempt body; if the cancel landed before the body ran, the
    # log should NOT contain the per-attempt phrase. The exact
    # substring "attempt" is too broad (it might appear in other
    # cluster-side log messages); pin instead on the cluster.connect
    # attempt-failed breadcrumb specifically.
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
