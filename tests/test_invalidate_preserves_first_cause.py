"""Repeated _invalidate calls must preserve the first cause, not overwrite
it with the synthetic "Not connected" wrapper from _ensure_connected,
which would bury the real transport error and harden leader-flip triage.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_invalidate_preserves_first_cause_across_repeats() -> None:
    """The first non-None cause sticks; subsequent invalidate calls do not overwrite it."""
    conn = DqliteConnection("localhost:9001")

    real_cause = OSError("connection reset by peer")
    conn._invalidate(real_cause)
    assert conn._invalidation_cause is real_cause

    synthetic = ConnectionError("synthetic wrapper")
    conn._invalidate(synthetic)
    assert conn._invalidation_cause is real_cause

    other = OSError("broken pipe")
    conn._invalidate(other)
    assert conn._invalidation_cause is real_cause


@pytest.mark.asyncio
async def test_invalidate_with_none_does_not_overwrite_first_cause() -> None:
    """cause=None on a subsequent invalidate must not clear a stored cause."""
    conn = DqliteConnection("localhost:9001")

    real_cause = OSError("first failure")
    conn._invalidate(real_cause)
    assert conn._invalidation_cause is real_cause

    conn._invalidate(None)
    assert conn._invalidation_cause is real_cause


@pytest.mark.asyncio
async def test_invalidate_first_cause_from_none_starts_record() -> None:
    """From None, the first non-None invalidate is the one that sticks."""
    conn = DqliteConnection("localhost:9001")

    assert conn._invalidation_cause is None
    conn._invalidate(None)
    assert conn._invalidation_cause is None

    real_cause = OSError("eventual failure")
    conn._invalidate(real_cause)
    assert conn._invalidation_cause is real_cause
