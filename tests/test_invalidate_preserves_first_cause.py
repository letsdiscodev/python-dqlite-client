"""Repeated _invalidate calls must preserve the first cause, not
overwrite with the synthetic "Not connected" wrapper that
_ensure_connected raises on subsequent operations against an
already-invalidated connection.

Without the guard, _invalidation_cause becomes a self-chaining stack of
"Not connected → Not connected → [real transport error]" wrappers. The
real cause is buried or dropped from the surface chain, making
leader-flip triage harder.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_invalidate_preserves_first_cause_across_repeats() -> None:
    """The first non-None cause sticks; subsequent invalidate calls do
    not overwrite it. This pins the root-cause-preservation contract
    used by operators reading invalidation chains during leader-flip
    triage."""
    conn = DqliteConnection("localhost:9001")

    real_cause = OSError("connection reset by peer")
    conn._invalidate(real_cause)
    assert conn._invalidation_cause is real_cause

    # A second invalidate with a synthetic wrapper must not overwrite
    # the original.
    synthetic = ConnectionError("synthetic wrapper")
    conn._invalidate(synthetic)
    assert conn._invalidation_cause is real_cause

    # Even a third one with a different transport-level error stays
    # pinned to the first cause; the chain is what callers should walk.
    other = OSError("broken pipe")
    conn._invalidate(other)
    assert conn._invalidation_cause is real_cause


@pytest.mark.asyncio
async def test_invalidate_with_none_does_not_overwrite_first_cause() -> None:
    """Passing cause=None on a subsequent invalidate must not clear or
    overwrite a previously-stored cause."""
    conn = DqliteConnection("localhost:9001")

    real_cause = OSError("first failure")
    conn._invalidate(real_cause)
    assert conn._invalidation_cause is real_cause

    conn._invalidate(None)
    assert conn._invalidation_cause is real_cause


@pytest.mark.asyncio
async def test_invalidate_first_cause_from_none_starts_record() -> None:
    """When _invalidation_cause starts as None, the first non-None
    invalidate is the one that sticks."""
    conn = DqliteConnection("localhost:9001")

    assert conn._invalidation_cause is None
    conn._invalidate(None)
    assert conn._invalidation_cause is None

    real_cause = OSError("eventual failure")
    conn._invalidate(real_cause)
    assert conn._invalidation_cause is real_cause
