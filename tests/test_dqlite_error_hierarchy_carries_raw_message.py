"""Pin: every client exception carries ``raw_message`` (default ``None``),
hoisted to the ``DqliteError`` base so verbatim server text survives
layer wrapping across the whole hierarchy, not just ``OperationalError``.
"""

from __future__ import annotations

from dqliteclient.exceptions import (
    ClusterError,
    ClusterPolicyError,
    DataError,
    DqliteConnectionError,
    DqliteError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)


def test_base_dqlite_error_has_raw_message_attribute() -> None:
    e = DqliteError("oops")
    assert e.raw_message is None


def test_base_dqlite_error_accepts_raw_message_kwarg() -> None:
    e = DqliteError("oops", raw_message="server text")
    assert e.raw_message == "server text"


def test_dqlite_connection_error_inherits_raw_message_from_base() -> None:
    e = DqliteConnectionError("Connection refused", raw_message="ECONNREFUSED")
    assert e.raw_message == "ECONNREFUSED"


def test_protocol_error_carries_raw_message() -> None:
    e = ProtocolError("Wire decode failed", raw_message="malformed frame")
    assert e.raw_message == "malformed frame"


def test_interface_error_carries_raw_message() -> None:
    e = InterfaceError("Connection is closed", raw_message="closed by peer")
    assert e.raw_message == "closed by peer"


def test_cluster_error_carries_raw_message() -> None:
    e = ClusterError("Could not find leader", raw_message="errors: ...")
    assert e.raw_message == "errors: ..."


def test_cluster_policy_error_carries_raw_message() -> None:
    e = ClusterPolicyError("rejected", raw_message="policy says no")
    assert e.raw_message == "policy says no"


def test_data_error_carries_raw_message() -> None:
    e = DataError("encode failed", raw_message="value too large")
    assert e.raw_message == "value too large"


def test_operational_error_keeps_existing_message_truncation_invariant() -> None:
    """OperationalError caps both message and raw_message (~4 KiB) so
    pickled exception graphs stay small under hostile-peer fan-out."""
    long = "X" * 5000
    e = OperationalError(long, 1, raw_message=long)
    assert "[truncated," in e.message
    assert len(e.raw_message) <= 4200
    assert "raw_message truncated" in e.raw_message


def test_operational_error_default_raw_message_is_message() -> None:
    """Backwards-compat: omitting raw_message= derives it from message."""
    e = OperationalError("boom", 1)
    assert e.raw_message == "boom"


def test_default_raw_message_is_none_for_other_classes() -> None:
    """Sibling classes default raw_message to None (no server text in scope)."""
    assert DqliteConnectionError("x").raw_message is None
    assert ProtocolError("x").raw_message is None
    assert InterfaceError("x").raw_message is None
    assert ClusterError("x").raw_message is None
    assert DataError("x").raw_message is None
