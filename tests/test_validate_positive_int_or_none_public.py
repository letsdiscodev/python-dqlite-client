"""``dqliteclient.validate_positive_int_or_none`` is a public re-export so
downstream packages do not reach into the private ``protocol`` symbol."""

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


def test_legacy_underscore_alias_removed() -> None:
    """The transitional underscore alias is removed; importing it must fail."""
    import dqliteclient.protocol as protocol_module

    assert not hasattr(protocol_module, "_validate_positive_int_or_none")
