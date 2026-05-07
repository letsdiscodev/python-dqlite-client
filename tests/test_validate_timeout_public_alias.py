"""Pin: ``validate_timeout`` is the public name for the timeout
validator and is re-exported from the package root. The
underscore-prefixed ``_validate_timeout`` is kept as a back-compat
alias to the same function so existing in-package call sites
continue to compile.

Downstream consumers (``dqlitedbapi.connection``) should import the
public name; cross-package imports of the underscore alias are a
PEP 8 non-public crossing.
"""

from __future__ import annotations

import pytest

import dqliteclient
from dqliteclient.connection import _validate_timeout, validate_timeout


def test_validate_timeout_in_public_all() -> None:
    assert "validate_timeout" in dqliteclient.__all__
    assert dqliteclient.validate_timeout is validate_timeout


def test_underscore_alias_identical_to_public() -> None:
    """``_validate_timeout`` is a direct alias to ``validate_timeout``;
    in-package call sites that still use the underscore name keep
    working without behaviour drift."""
    assert _validate_timeout is validate_timeout


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
