"""Pin: ``DqliteConnection._close_impl`` reaps a fresh
``_pending_drain`` task that a concurrent ``_invalidate`` callback
created during the prior drain's ``await pending``.

Without the bounded re-snapshot loop, a
``loop.call_soon_threadsafe(self._invalidate, ...)`` callback queued
by the dbapi sync wrapper's timeout / KI arms can run during
``await pending``, find ``self._protocol`` still non-None, and create
a NEW ``_pending_drain`` task. After the original ``await pending``
resolves, ``_close_impl`` reads ``self._protocol`` (now None — the
racing ``_invalidate`` cleared it), short-circuits, and returns —
leaving the new drain task orphaned. asyncio emits
"Task was destroyed but it is pending" at interpreter shutdown.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from dqliteclient.connection import DqliteConnection


@pytest.mark.asyncio
async def test_close_impl_drains_late_arriving_pending_drain() -> None:
    """Drive the race shape: snapshot pending=A; during ``await A``,
    a callback creates pending=B (a still-running task). The bounded
    re-snapshot loop must observe B and await it, leaving no orphan.
    """
    loop = asyncio.get_running_loop()

    async def _slow_drain(label: str) -> None:
        # Survive long enough that close_impl observes us as not done.
        await asyncio.sleep(0.005)

    pending_a = loop.create_task(_slow_drain("A"))
    pending_b_holder: list[asyncio.Task[None]] = []

    def _race_invalidate_callback() -> None:
        # Simulates the dbapi sync wrapper's
        # ``loop.call_soon_threadsafe(dying._invalidate, ...)`` that
        # runs while close_impl is awaiting the prior drain.
        b = loop.create_task(_slow_drain("B"))
        pending_b_holder.append(b)
        conn._pending_drain = b

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = pending_a
    conn._protocol = None  # short-circuit the rest of close_impl
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop = None

    # Schedule the race callback to fire while we're inside the
    # await pending block. ``call_soon`` runs at the next loop
    # iteration which is exactly when ``await pending`` yields.
    loop.call_soon(_race_invalidate_callback)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        await conn._close_impl()
        # Let the loop drain so any orphaned task surfaces as a
        # "Task was destroyed but it is pending" warning at GC.
        await asyncio.sleep(0.05)

    # Both A and B must be done — the loop reaped both.
    assert pending_a.done()
    assert pending_b_holder, "race callback should have run"
    assert pending_b_holder[0].done(), (
        "fresh _pending_drain task created during await must be reaped"
    )
    # No "Task was destroyed but it is pending" warnings on the
    # warnings channel.
    asyncio_warnings = [w for w in captured if "Task was destroyed" in str(w.message)]
    assert not asyncio_warnings, (
        f"Expected no orphaned-task warnings; got {[str(w.message) for w in asyncio_warnings]}"
    )


@pytest.mark.asyncio
async def test_close_impl_loop_terminates_when_invalidate_clears_protocol() -> None:
    """The re-snapshot loop terminates after one iteration when the
    racing ``_invalidate`` clears ``_protocol`` (the production
    callback path) — no infinite spin."""
    loop = asyncio.get_running_loop()

    async def _slow(label: str) -> None:
        await asyncio.sleep(0.001)

    pending_a = loop.create_task(_slow("A"))

    conn = DqliteConnection.__new__(DqliteConnection)
    conn._pending_drain = pending_a
    conn._protocol = None
    conn._in_transaction = False
    conn._tx_owner = None
    conn._savepoint_stack = []
    conn._savepoint_implicit_begin = False
    conn._has_untracked_savepoint = False
    conn._invalidation_cause = None
    conn._bound_loop = None

    await conn._close_impl()
    assert pending_a.done()
    assert conn._pending_drain is None
