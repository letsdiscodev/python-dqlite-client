"""Pin: ``ConnectionPool.close()`` fork short-circuit runs BEFORE the
``_closed`` early-return so a child whose parent forked mid-close
(with ``_close_done`` set but not yet ``set()``) does not hang on
a parent-loop Event.

The fork-after-init guard short-circuits ``close()`` to a local-state
flip without touching inherited FDs. The pid check has to run BEFORE
the ``if self._closed:`` arm that awaits ``_close_done`` — otherwise a
child that fell into the "second-caller-waits" arm would await an
``asyncio.Event`` bound to the parent's defunct event loop, blocking
forever in the child's fresh loop.

Reproduce by simulating a fork-during-close: stage a pool where
``_closed=True`` and ``_close_done`` is an Event bound to a different
loop, then call ``close()`` from a fresh loop with a fake-mismatched
pid, and confirm it returns immediately instead of hanging.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.pool import ConnectionPool


def _make_pool() -> ConnectionPool:
    cluster = MagicMock(spec=ClusterClient)
    return ConnectionPool(
        addresses=["localhost:9001"],
        min_size=0,
        max_size=1,
        timeout=1.0,
        cluster=cluster,
    )


@pytest.mark.asyncio
async def test_close_with_pid_mismatch_returns_even_when_closed_with_pending_event() -> None:
    """``close()`` in a forked child must short-circuit to a local-
    state flip and return immediately, even if ``_closed=True`` and
    ``_close_done`` is set but not yet completed (i.e. the parent's
    ``_drain_idle`` was in flight at fork time).

    Without the fix the child enters the ``if self._closed:`` arm
    and awaits ``_close_done.wait()`` forever, because the Event was
    bound to the parent's loop and will never be set in the child.
    """
    pool = _make_pool()
    pool._closed = True
    # Stage an Event the child would inherit — bound to *our* loop, but
    # we will pretend it is the parent's via the pid mismatch. The
    # important property is that nobody will ``set()`` it during this
    # test, so a hang would be observable as a TimeoutError.
    pool._close_done = asyncio.Event()
    fake_parent_pid = pool._creator_pid + 1
    pool._creator_pid = fake_parent_pid  # Simulate "we are the child."

    # Bound the await so a regression manifests as a clean failure
    # instead of a hung test.
    with patch("dqliteclient.pool.os.getpid", return_value=fake_parent_pid + 1):
        await asyncio.wait_for(pool.close(), timeout=1.0)

    # Local state flipped, no wait happened.
    assert pool._closed is True


@pytest.mark.asyncio
async def test_close_with_pid_mismatch_returns_even_when_closed_with_no_event() -> None:
    """Sibling case: ``_closed=True`` but ``_close_done`` was never
    published (synchronous-close-and-set window). Child must still
    short-circuit and return — without entering the ``if
    self._closed:`` arm at all."""
    pool = _make_pool()
    pool._closed = True
    pool._close_done = None
    fake_parent_pid = pool._creator_pid + 1
    pool._creator_pid = fake_parent_pid

    with patch("dqliteclient.pool.os.getpid", return_value=fake_parent_pid + 1):
        await asyncio.wait_for(pool.close(), timeout=1.0)

    assert pool._closed is True
