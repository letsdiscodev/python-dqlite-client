"""Pin: ``CLOSE_TIMEOUT_FLOOR_RATIONALE`` is a public, importable
constant — both via the underlying module and via the package
top-level surface.

The constant carries the operator-facing FIN-flush / TIME_WAIT
explanation that every ``close_timeout`` validator caller appends to
the floor-rejection diagnostic. Since it is consumed cross-package
(by python-dqlite-dbapi's ``_validate_close_timeout`` wrap), the
underscore-prefixed shape was a brittle private contract; the public
shape mirrors the ``_sanitize_for_log`` -> ``sanitize_for_log``
promotion pattern.
"""

from __future__ import annotations

import dqliteclient
from dqliteclient.connection import CLOSE_TIMEOUT_FLOOR_RATIONALE


def test_constant_present_at_package_top_level() -> None:
    assert hasattr(dqliteclient, "CLOSE_TIMEOUT_FLOOR_RATIONALE")
    assert dqliteclient.CLOSE_TIMEOUT_FLOOR_RATIONALE is CLOSE_TIMEOUT_FLOOR_RATIONALE


def test_constant_in_package_all_export_list() -> None:
    assert "CLOSE_TIMEOUT_FLOOR_RATIONALE" in dqliteclient.__all__


def test_constant_carries_fin_flush_text() -> None:
    """The text is the operator-facing rationale; FIN-flush language
    is load-bearing for a downstream regression test in the dbapi
    layer that asserts the rationale reaches the user."""
    assert "FIN flushes" in CLOSE_TIMEOUT_FLOOR_RATIONALE
    assert "TIME_WAIT" in CLOSE_TIMEOUT_FLOOR_RATIONALE


def test_underscore_alias_removed() -> None:
    """The previous underscore-prefixed name should no longer be
    importable — a hard removal forces downstream consumers to adopt
    the public shape rather than relying on a transitional alias."""
    import dqliteclient.connection as _conn_mod

    assert not hasattr(_conn_mod, "_CLOSE_TIMEOUT_FLOOR_RATIONALE"), (
        "_CLOSE_TIMEOUT_FLOOR_RATIONALE underscore alias must be removed"
    )
