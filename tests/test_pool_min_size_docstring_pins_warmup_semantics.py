"""Pin: the ``min_size`` docstring on :class:`ConnectionPool` and
on :func:`dqliteclient.connect` documents the warm-up (not
steady-state floor) semantics.

The verb "maintain" implied an active-refill contract that the
implementation does not honour. ``initialize()`` is the only site
that reads ``self._min_size`` for pre-creation; ``_drain_idle``
sweeps from the dead-conn / broken-conn arms in ``acquire()``
contract the queue past ``min_size`` and the pool does not refill.
asyncpg refills via ``release(terminate=True)``; go-dqlite refills
via its pool-maintainer goroutine; this pool does neither.

A future contributor scanning the docstring and tightening the
semantics to active refill would land a real new feature with
subtle interactions (a refill task running concurrently with
``close()`` would need shield discipline). Pin the warm-up
disclaimer so the contract stays honest until that work is
deliberately undertaken.
"""

from __future__ import annotations

from dqliteclient import create_pool
from dqliteclient.pool import ConnectionPool


def test_pool_min_size_docstring_documents_warmup_semantics() -> None:
    doc = ConnectionPool.__init__.__doc__ or ""
    assert "pre-warm" in doc, (
        "ConnectionPool docstring must name min_size as a pre-warm count, not a steady-state floor"
    )
    assert "NOT a steady-state floor" in doc, (
        "ConnectionPool docstring must explicitly disclaim active refill "
        "so future contributors don't tighten the contract silently"
    )


def test_create_pool_factory_min_size_docstring_cross_refs_pool() -> None:
    doc = create_pool.__doc__ or ""
    assert "pre-warm" in doc, (
        "The top-level ``create_pool`` factory must mirror the "
        "ConnectionPool wording so URL-mode and pool-mode users see the "
        "same contract"
    )
