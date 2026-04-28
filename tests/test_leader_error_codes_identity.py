"""Pin: ``LEADER_ERROR_CODES`` is the single source of truth for the
leader-change SQLite extended codes, owned by ``dqlitewire`` and
imported as-is by ``dqliteclient``. Identity equality is the
contract — a future SA-only override or accidental local copy would
fail this test.
"""

from __future__ import annotations


def test_dqliteclient_uses_wire_leader_error_codes_identity() -> None:
    # The constants live inside each module but are deliberately NOT
    # in ``__all__`` (they're re-imports of a wire-layer public
    # constant, not the module's own public surface). Direct
    # attribute access works at runtime; the mypy ignores reflect
    # the re-import-without-export pattern.
    import dqliteclient.connection as _conn_mod
    import dqliteclient.pool as _pool_mod
    from dqlitewire import LEADER_ERROR_CODES as wire_codes

    assert _pool_mod.LEADER_ERROR_CODES is wire_codes  # type: ignore[attr-defined]
    assert _conn_mod.LEADER_ERROR_CODES is wire_codes  # type: ignore[attr-defined]
