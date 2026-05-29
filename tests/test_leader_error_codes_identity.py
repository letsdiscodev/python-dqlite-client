"""Pin: ``dqliteclient`` imports ``dqlitewire.LEADER_ERROR_CODES`` by identity, not copy."""

from __future__ import annotations


def test_dqliteclient_uses_wire_leader_error_codes_identity() -> None:
    # Deliberately not in ``__all__`` (re-imports, not own surface); mypy ignores cover that.
    import dqliteclient.connection as _conn_mod
    import dqliteclient.pool as _pool_mod
    from dqlitewire import LEADER_ERROR_CODES as wire_codes

    assert _pool_mod.LEADER_ERROR_CODES is wire_codes  # type: ignore[attr-defined]
    assert _conn_mod.LEADER_ERROR_CODES is wire_codes  # type: ignore[attr-defined]
