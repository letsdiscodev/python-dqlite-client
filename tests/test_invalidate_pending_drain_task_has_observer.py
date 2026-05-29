"""Pin: ``_invalidate``'s fresh ``_pending_drain`` task gets
``_observe_drain_exception`` attached too, so a BaseException escape from
``_bounded_drain`` is not left unobserved at GC.
"""

from __future__ import annotations

import inspect

from dqliteclient.connection import DqliteConnection


def test_invalidate_attaches_drain_observer_to_fresh_pending_drain() -> None:
    """``_invalidate`` must attach ``_observe_drain_exception`` to the fresh
    ``_pending_drain`` task, not only the prior one."""
    src = inspect.getsource(DqliteConnection._invalidate)
    callback_attachment_count = src.count("add_done_callback(_observe_drain_exception)")
    assert callback_attachment_count >= 2, (
        "Expected both the prior-task and fresh-task code paths in "
        "``_invalidate`` to call ``add_done_callback("
        "_observe_drain_exception)``; found only "
        f"{callback_attachment_count} site(s)."
    )
