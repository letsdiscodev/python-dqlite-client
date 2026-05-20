"""Pin: ``DqliteConnection._invalidate``'s fresh ``_pending_drain``
task has ``_observe_drain_exception`` attached, symmetric with the
cancel-and-detach idiom for the prior task.

Without the observer, a ``BaseException``-class escape from
``_bounded_drain``'s ``suppress(Exception)`` (programmer-bug paths
inside the suppress contract, or ``KeyboardInterrupt`` /
``SystemExit`` mid-drain) would leave the task unobserved at GC and
asyncio's task finaliser would emit "Task exception was never
retrieved" with no actionable context.
"""

from __future__ import annotations

import inspect

from dqliteclient.connection import DqliteConnection


def test_invalidate_attaches_drain_observer_to_fresh_pending_drain() -> None:
    """Inspection pin: ``_invalidate`` must call
    ``add_done_callback(_observe_drain_exception)`` on the fresh
    ``_pending_drain`` task, not only on the prior one."""
    src = inspect.getsource(DqliteConnection._invalidate)
    # Two add_done_callback sites expected: one for the prior task
    # (cancel-and-detach idiom) and one for the fresh task.
    callback_attachment_count = src.count("add_done_callback(_observe_drain_exception)")
    assert callback_attachment_count >= 2, (
        "Expected both the prior-task and fresh-task code paths in "
        "``_invalidate`` to call ``add_done_callback("
        "_observe_drain_exception)``; found only "
        f"{callback_attachment_count} site(s)."
    )
