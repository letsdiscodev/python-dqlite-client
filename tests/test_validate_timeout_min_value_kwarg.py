"""Pin: ``validate_timeout(value, min_value=...)`` enforces the floor
uniformly across all callers, including direct ``DqliteConnection`` /
``ConnectionPool`` constructors that bypass the dbapi layer.

Threat model: a programmer using ``dqliteclient.DqliteConnection(addr,
close_timeout=0.001)`` directly bypasses the SA / dbapi-layer floor.
Below 0.01 s the dispose-time writer-close may complete before FIN
flushes, leaving connections lingering in TIME_WAIT. The dbapi-side
floor lives at the client layer's ``validate_timeout`` so every
entry path enforces the same contract.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import DqliteConnection, validate_timeout
from dqliteclient.pool import ConnectionPool


def test_validate_timeout_default_min_value_zero_exclusive() -> None:
    validate_timeout(0.001, name="timeout")  # accepted (default floor is 0)


def test_validate_timeout_below_explicit_min_value_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0.01"):
        validate_timeout(0.001, name="close_timeout", min_value=0.01)


def test_validate_timeout_at_explicit_min_value_accepted() -> None:
    validate_timeout(0.01, name="close_timeout", min_value=0.01)


def test_validate_timeout_above_explicit_min_value_accepted() -> None:
    validate_timeout(0.5, name="close_timeout", min_value=0.01)


def test_validate_timeout_zero_rejected_before_min_value_check() -> None:
    """Zero is rejected by the inner positive check before the
    min_value floor — diagnostic mentions positivity, not the floor."""
    with pytest.raises(ValueError, match="positive finite number"):
        validate_timeout(0.0, name="close_timeout", min_value=0.01)


def test_validate_timeout_floor_diagnostic_omits_fin_text_without_rationale() -> None:
    """The generic validator no longer hard-codes the FIN-flush
    rationale: callers that pass ``min_value=`` for non-close-timeout
    reasons would otherwise inherit a misleading explanation. Without
    a ``min_value_rationale=`` kwarg, the diagnostic stays neutral."""
    with pytest.raises(ValueError, match=r"close_timeout must be >= 0\.01") as exc:
        validate_timeout(0.001, name="close_timeout", min_value=0.01)
    assert "FIN flushes" not in str(exc.value)
    assert "TIME_WAIT" not in str(exc.value)


def test_validate_timeout_message_mentions_fin_flush_rationale_when_passed() -> None:
    """When the caller supplies ``min_value_rationale=``, the
    explanation is appended to the diagnostic. The close_timeout
    callers pass the FIN-flush rationale."""
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
    """Direct DqliteConnection caller bypasses the dbapi-layer floor;
    the client-layer enforcement now catches them — and the constructor
    passes the FIN-flush rationale so the diagnostic includes the
    operator-facing explanation."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01") as exc:
        DqliteConnection("localhost:9001", close_timeout=0.001)
    assert "FIN flushes" in str(exc.value), (
        "DqliteConnection must wrap validate_timeout with the "
        "close-timeout-specific FIN-flush rationale so operators "
        "understand the reason for the floor."
    )


def test_connection_pool_close_timeout_below_floor_rejected() -> None:
    """Same for ConnectionPool — also wraps with the FIN-flush rationale."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01") as exc:
        ConnectionPool(addresses=["localhost:9001"], close_timeout=0.001)
    assert "FIN flushes" in str(exc.value)


def test_dqlite_connection_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works (regression)."""
    DqliteConnection("localhost:9001", close_timeout=0.5)


def test_connection_pool_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works (regression)."""
    ConnectionPool(addresses=["localhost:9001"], close_timeout=0.5)
