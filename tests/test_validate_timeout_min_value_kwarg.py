"""Pin: ``validate_timeout(value, min_value=...)`` enforces the floor
uniformly across all callers, including direct ``DqliteConnection`` /
``ConnectionPool`` constructors that bypass the dbapi layer.

Threat model: a programmer using ``dqliteclient.DqliteConnection(addr,
close_timeout=0.001)`` directly bypasses the SA / dbapi-layer floor.
Below 0.01 s the dispose-time writer-close may complete before FIN
flushes, leaving connections lingering in TIME_WAIT. The dbapi-side
floor was added in a prior round; this round centralises it at the
client layer's ``validate_timeout`` so every entry path enforces the
same contract.
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


def test_validate_timeout_message_mentions_fin_flush_rationale() -> None:
    """Floor-rejection diagnostic explains the TIME_WAIT rationale so
    operators understand why the floor exists."""
    with pytest.raises(ValueError, match="FIN flushes"):
        validate_timeout(0.001, name="close_timeout", min_value=0.01)


def test_dqlite_connection_close_timeout_below_floor_rejected() -> None:
    """Direct DqliteConnection caller bypasses the dbapi-layer floor;
    the client-layer enforcement now catches them."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01"):
        DqliteConnection("localhost:9001", close_timeout=0.001)


def test_connection_pool_close_timeout_below_floor_rejected() -> None:
    """Same for ConnectionPool."""
    with pytest.raises(ValueError, match="close_timeout must be >= 0.01"):
        ConnectionPool(addresses=["localhost:9001"], close_timeout=0.001)


def test_dqlite_connection_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works (regression)."""
    DqliteConnection("localhost:9001", close_timeout=0.5)


def test_connection_pool_close_timeout_at_default_accepted() -> None:
    """Default close_timeout=0.5 still works (regression)."""
    ConnectionPool(addresses=["localhost:9001"], close_timeout=0.5)
