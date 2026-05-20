"""Pin: every ``add_done_callback``-reachable exception-observation
site in the client and dbapi-async layers uses
``contextlib.suppress(Exception)`` rather than the wider
``BaseException``.

The project established a uniform narrow-suppress discipline across
``_close_impl`` (ISSUE-1198), ``cluster.py`` close-after-shielded
(ISSUE-982), pool release-shielded-pending-drain (ISSUE-787), SA
``terminate`` close-arm (ISSUE-882): never suppress
``KeyboardInterrupt`` / ``SystemExit`` raised by a signal handler
at a bytecode boundary inside the suppressed body. Three sites
survived the prior sweep:

* ``_observe_drain_exception`` in cluster.py.
* ``_clear_slot``'s inline observe in cluster.py (duplicated body).
* dbapi-async ``force_close_transport``'s ``_cancel_and_observe``
  closure.

Inspection pin (source-substring) rather than runtime pin because
the failure mode is a signal-driven KI / SystemExit at a specific
bytecode boundary inside ``t.exception()`` — not reproducible in a
single-process test without forking a signal-source process.
"""

from __future__ import annotations

import inspect


def test_observe_drain_exception_uses_narrow_suppress() -> None:
    """``_observe_drain_exception`` suppresses ``Exception``, not
    ``BaseException``, so a signal-driven KI / SystemExit inside the
    ``t.exception()`` call still propagates."""
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
    """The inline ``_clear_slot`` closure inside ``find_leader``
    duplicates the observe-pattern and must follow the same
    discipline. Inspected via the enclosing method source."""
    from dqliteclient.cluster import ClusterClient

    src = inspect.getsource(ClusterClient.find_leader)
    assert "contextlib.suppress(BaseException)" not in src, (
        "_clear_slot's done-callback observe must use narrow "
        "Exception suppression — see _observe_drain_exception sibling"
    )
    # The inline observe is only one of several site shapes inside
    # find_leader; assert the narrow form appears at least once.
    assert "contextlib.suppress(Exception)" in src
