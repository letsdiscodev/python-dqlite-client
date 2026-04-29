"""Pin: ``find_leader`` registers its done-callback BEFORE inserting
the task into the shared slot map.

The reverse order (slot insert → add_done_callback) opens a
race window: a signal-driven interrupt (KeyboardInterrupt /
SystemExit raised by a signal handler at any bytecode boundary
between the two statements) leaves the slot pointing at a task
whose completion is never observed by ``_clear_slot``.

This test inspects the source order via the closure-captured
function bytecode order; a future refactor that reverts the
ordering will fail this pin.
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def test_done_callback_registered_before_slot_insert() -> None:
    """Inspect the find_leader source: ``add_done_callback`` must
    appear before ``self._find_leader_tasks[key] = task`` in the
    ``if task is None or task.done():`` branch."""
    source = inspect.getsource(ClusterClient.find_leader)
    callback_idx = source.find("add_done_callback")
    slot_assign_idx = source.find("self._find_leader_tasks[key] = task")
    assert callback_idx > 0, "add_done_callback registration site must exist"
    assert slot_assign_idx > 0, "slot insert site must exist"
    assert callback_idx < slot_assign_idx, (
        "find_leader must register the done-callback before inserting "
        "the task into the shared slot map; otherwise a signal-driven "
        "interrupt between the two statements leaks an unobserved-"
        "exception task"
    )
