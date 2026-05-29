"""``validate_timeout(value, min_value=...)`` enforces the floor on every
entry path, including direct DqliteConnection / ConnectionPool constructors.
Below the 0.01 s close_timeout floor the dispose-time writer-close may finish
before FIN flushes, leaving connections in TIME_WAIT."""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection, validate_timeout
from dqliteclient.pool import ConnectionPool


def test_validate_timeout_default_min_value_zero_exclusive() -> None:
    validate_timeout(0.001, name="timeout")  # default floor is 0


def test_validate_timeout_below_explicit_min_value_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0.01"):
        validate_timeout(0.001, name="close_timeout", min_value=0.01)


def test_validate_timeout_at_explicit_min_value_accepted() -> None:
    validate_timeout(0.01, name="close_timeout", min_value=0.01)


def test_validate_timeout_above_explicit_min_value_accepted() -> None:
    validate_timeout(0.5, name="close_timeout", min_value=0.01)


def test_validate_timeout_zero_rejected_before_min_value_check() -> None:
    """Zero hits the positive check before the floor; diagnostic mentions
    positivity, not the floor."""
    with pytest.raises(ValueError, match="positive finite number"):
        validate_timeout(0.0, name="close_timeout", min_value=0.01)


def test_validate_timeout_floor_diagnostic_omits_fin_text_without_rationale() -> None:
    """Without ``min_value_rationale=`` the diagnostic stays neutral, so
    non-close-timeout callers do not inherit the FIN-flush explanation."""
    with pytest.raises(ValueError, match=r"close_timeout must be >= 0\.01") as exc:
        validate_timeout(0.001, name="close_timeout", min_value=0.01)
    assert "FIN flushes" not in str(exc.value)
    assert "TIME_WAIT" not in str(exc.value)


def test_validate_timeout_message_mentions_fin_flush_rationale_when_passed() -> None:
    """``min_value_rationale=`` is appended to the diagnostic."""
    with pytest.raises(ValueError, match="FIN flushes"):
        validate_timeout(
            0.001,
            name="close_timeout",
            min_value=0.01,
            min_value_rationale=(
                "Below this floor, the dispose-time writer-close may "
                "complete before FIN flushes, leaving connections "
                "lingering in TIME_WAIT."
            ),
        )


def test_dqlite_connection_close_timeout_below_floor_rejected() -> None:
    """The constructor enforces the floor and passes the FIN-flush rationale."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01") as exc:
        DqliteConnection("localhost:9001", close_timeout=0.001)
    assert "FIN flushes" in str(exc.value), (
        "DqliteConnection must wrap validate_timeout with the "
        "close-timeout-specific FIN-flush rationale so operators "
        "understand the reason for the floor."
    )


def test_connection_pool_close_timeout_below_floor_rejected() -> None:
    """Same for ConnectionPool — also passes the FIN-flush rationale."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01") as exc:
        ConnectionPool(addresses=["localhost:9001"], close_timeout=0.001)
    assert "FIN flushes" in str(exc.value)


def test_dqlite_connection_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works."""
    DqliteConnection("localhost:9001", close_timeout=0.5)


def test_connection_pool_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works."""
    ConnectionPool(addresses=["localhost:9001"], close_timeout=0.5)
