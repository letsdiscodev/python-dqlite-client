"""``CLOSE_TIMEOUT_FLOOR_RATIONALE`` must be a public, importable constant (module and
package top-level): it is consumed cross-package by python-dqlite-dbapi's
``_validate_close_timeout``, so the underscore-prefixed shape was a brittle private
contract."""

from __future__ import annotations

import dqliteclient
from dqliteclient.connection import CLOSE_TIMEOUT_FLOOR_RATIONALE


def test_constant_present_at_package_top_level() -> None:
    assert hasattr(dqliteclient, "CLOSE_TIMEOUT_FLOOR_RATIONALE")
    assert dqliteclient.CLOSE_TIMEOUT_FLOOR_RATIONALE is CLOSE_TIMEOUT_FLOOR_RATIONALE


def test_constant_in_package_all_export_list() -> None:
    assert "CLOSE_TIMEOUT_FLOOR_RATIONALE" in dqliteclient.__all__


def test_constant_carries_fin_flush_text() -> None:
    """FIN-flush language is load-bearing for a downstream dbapi test asserting the
    rationale reaches the user."""
    assert "FIN flushes" in CLOSE_TIMEOUT_FLOOR_RATIONALE
    assert "TIME_WAIT" in CLOSE_TIMEOUT_FLOOR_RATIONALE


def test_underscore_alias_removed() -> None:
    """The underscore-prefixed name must no longer be importable (no transitional alias)."""
    import dqliteclient.connection as _conn_mod

    assert not hasattr(_conn_mod, "_CLOSE_TIMEOUT_FLOOR_RATIONALE"), (
        "_CLOSE_TIMEOUT_FLOOR_RATIONALE underscore alias must be removed"
    )
