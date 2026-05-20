"""Pin: ``_connect_impl`` uses a bounded re-snapshot loop (cap=3)
to retire ``_pending_drain``, mirroring ``_close_impl``'s
discipline.

A racing ``_invalidate`` scheduled via ``call_soon_threadsafe``
from the dbapi-sync wrapper's timeout / KI arms can publish a FRESH
``_pending_drain`` task during ``await pending``. The previous
single-shot ``self._pending_drain = None`` at the end of the
retire logic would null the fresh task, orphaning it on the loop
(``"Task was destroyed but it is pending"`` at GC).

Inspection pins on the source — the runtime path requires
``call_soon_threadsafe``-injecting a fresh task mid-await, which
is invasive to set up reliably in a unit test.
"""

from __future__ import annotations

import inspect

from dqliteclient.connection import DqliteConnection


def test_connect_impl_uses_bounded_resnapshot_loop() -> None:
    """``_connect_impl`` must carry the same re-snapshot loop shape
    as ``_close_impl`` so a racing ``_invalidate`` cannot orphan a
    fresh drain task."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    assert "resnapshot_cap = 3" in src, (
        "connect-side re-snapshot loop must use the same cap as close"
    )
    assert "for _attempt in range(resnapshot_cap):" in src
    # Cap-exhausted arm attaches the observer (mirror of close).
    assert "_observe_drain_exception" in src
    assert "stuck.cancel()" in src


def test_connect_impl_cap_exhausted_arm_logs_warning() -> None:
    """The pathological feedback-loop case must surface in
    production logs."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    assert "DqliteConnection._connect_impl: _pending_drain still set after" in src
    assert "feedback loop on connection id" in src


def test_connect_impl_preserves_cancelling_delta_pattern() -> None:
    """The unique connect-side cancelling-delta dance — detect outer
    cancel via ``Task.cancelling()`` counter delta — must survive
    the refactor. Without it, ``connect()`` would silently swallow
    an outer cancel and open a TCP connection the caller intended
    to abort."""
    src = inspect.getsource(DqliteConnection._connect_impl)
    assert "cancelling_before" in src
    assert "cancelling_after" in src
    assert "cancelling_after > cancelling_before" in src


def test_close_impl_and_connect_impl_share_resnapshot_cap_value() -> None:
    """Cross-method symmetry: the cap value matches so a regression
    that bumps one but not the other trips this pin."""
    connect_src = inspect.getsource(DqliteConnection._connect_impl)
    close_src = inspect.getsource(DqliteConnection._close_impl)
    assert "resnapshot_cap = 3" in connect_src
    assert "resnapshot_cap = 3" in close_src
