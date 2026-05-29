"""``validate_timeout`` is the public name re-exported from the package root;
the old ``_validate_timeout`` alias has been removed."""

from __future__ import annotations

import pytest

import dqliteclient
from dqliteclient.connection import validate_timeout


def test_validate_timeout_in_public_all() -> None:
    assert "validate_timeout" in dqliteclient.__all__
    assert dqliteclient.validate_timeout is validate_timeout


def test_underscore_alias_removed_from_module() -> None:
    """The client ``_validate_timeout`` alias is deleted (the dbapi has an
    unrelated same-named symbol on its own module)."""
    import dqliteclient.connection as _conn_mod

    assert not hasattr(_conn_mod, "_validate_timeout")


def test_validate_timeout_accepts_positive_finite() -> None:
    assert validate_timeout(1.0) == 1.0
    assert validate_timeout(0.5, name="dial_timeout") == 0.5


def test_validate_timeout_rejects_bool_zero_negative_nonfinite() -> None:
    with pytest.raises(ValueError, match="bool"):
        validate_timeout(True)
    with pytest.raises(ValueError, match="positive"):
        validate_timeout(0)
    with pytest.raises(ValueError, match="positive"):
        validate_timeout(-1.0)
    with pytest.raises(ValueError, match="positive"):
        validate_timeout(float("inf"))
    with pytest.raises(TypeError, match="number"):
        validate_timeout("1.0")  # type: ignore[arg-type]
