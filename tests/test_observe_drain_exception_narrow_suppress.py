"""Done-callback observe sites suppress ``Exception``, not ``BaseException``, so a
signal-driven KeyboardInterrupt / SystemExit still propagates. Source-substring pin: the
failure mode (signal at a bytecode boundary inside ``t.exception()``) is not reproducible
single-process."""

from __future__ import annotations

import inspect


def test_observe_drain_exception_uses_narrow_suppress() -> None:
    from dqliteclient.cluster import _observe_drain_exception

    src = inspect.getsource(_observe_drain_exception)
    assert "contextlib.suppress(Exception)" in src
    assert "contextlib.suppress(BaseException)" not in src, (
        "_observe_drain_exception must narrow-suppress only Exception; "
        "BaseException width swallows KeyboardInterrupt / SystemExit "
        "raised in signal-handler windows. See project-wide narrow-"
        "suppress discipline established at _close_impl / pool / SA."
    )


def test_find_leader_clear_slot_callback_uses_narrow_suppress() -> None:
    """The inline ``_clear_slot`` closure in ``find_leader`` must follow the same discipline."""
    from dqliteclient.cluster import ClusterClient

    src = inspect.getsource(ClusterClient.find_leader)
    assert "contextlib.suppress(BaseException)" not in src, (
        "_clear_slot's done-callback observe must use narrow "
        "Exception suppression — see _observe_drain_exception sibling"
    )
    assert "contextlib.suppress(Exception)" in src
