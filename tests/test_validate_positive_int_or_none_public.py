"""Pin: ``dqliteclient.validate_positive_int_or_none`` is a public
re-export so downstream packages (dqlitedbapi, sqlalchemy-dqlite) do
not need to reach into private symbols.

The validator was previously available only as
``dqliteclient.protocol._validate_positive_int_or_none`` (leading
underscore, not in any ``__all__``). dqlitedbapi imported it directly,
creating a cross-package coupling that a future client refactor could
silently break. Promote to public, mirror the ``parse_address`` /
``allowlist_policy`` pattern, and pin the public surface so it cannot
disappear without a breaking-change signal.
"""

from __future__ import annotations

import pytest

import dqliteclient


def test_public_validate_positive_int_or_none_callable() -> None:
    assert callable(dqliteclient.validate_positive_int_or_none)


def test_public_name_in_all() -> None:
    assert "validate_positive_int_or_none" in dqliteclient.__all__


def test_public_validator_accepts_positive_int() -> None:
    assert dqliteclient.validate_positive_int_or_none(1, "x") == 1
    assert dqliteclient.validate_positive_int_or_none(10**6, "x") == 10**6


def test_public_validator_accepts_none() -> None:
    assert dqliteclient.validate_positive_int_or_none(None, "x") is None


def test_public_validator_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        dqliteclient.validate_positive_int_or_none(0, "x")
    with pytest.raises(ValueError, match="must be > 0"):
        dqliteclient.validate_positive_int_or_none(-1, "x")


def test_public_validator_rejects_bool() -> None:
    with pytest.raises(TypeError, match="must be int or None"):
        dqliteclient.validate_positive_int_or_none(True, "x")


def test_public_validator_rejects_non_int() -> None:
    with pytest.raises(TypeError, match="must be int or None"):
        dqliteclient.validate_positive_int_or_none(1.5, "x")  # type: ignore[arg-type]


def test_legacy_underscore_alias_still_works() -> None:
    """The leading-underscore alias is preserved for one release as a
    deprecation cushion. Pin so a future deletion of the alias is an
    explicit, reviewed change."""
    from dqliteclient.protocol import _validate_positive_int_or_none

    assert _validate_positive_int_or_none is dqliteclient.validate_positive_int_or_none
