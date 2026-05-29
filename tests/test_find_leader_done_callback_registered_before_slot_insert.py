"""``find_leader`` must register its done-callback before inserting the
task into the shared slot map; the reverse order lets a signal-driven
interrupt leave the slot pointing at a task ``_clear_slot`` never observes."""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def test_done_callback_registered_before_slot_insert() -> None:
    """``add_done_callback`` must appear before the slot assignment in
    source."""
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
