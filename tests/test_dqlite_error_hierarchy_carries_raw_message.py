"""Pin: every client exception class carries the ``raw_message``
attribute (defaulting to ``None``), so the cycle-21 invariant — the
verbatim server text survives layer wrapping — applies symmetrically
across the hierarchy and not only to ``OperationalError``.

The dbapi-layer ``getattr(e, "raw_message", None) or str(e)`` idiom
previously documented its fallback as "older client versions without
the attribute"; that rationale was a fiction (the current client
lacked the attribute too). After hoisting ``raw_message`` to the
``DqliteError`` base, every subclass exposes the attribute as a
property and downstream consumers can read it without ``getattr``.
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


def test_operational_error_keeps_existing_raw_message_truncation_invariant() -> None:
    """OperationalError still truncates ``message`` for display while
    keeping the unbounded server text on ``raw_message`` — the
    original cycle-21 contract is not broken by the hoist."""
    long = "X" * 5000
    e = OperationalError(1, long, raw_message=long)
    assert "[truncated," in e.message
    assert e.raw_message == long  # untruncated


def test_operational_error_default_raw_message_is_message() -> None:
    """Backwards-compat: if ``raw_message=`` is omitted, OperationalError
    derives it from ``message`` per the existing contract."""
    e = OperationalError(1, "boom")
    assert e.raw_message == "boom"


def test_default_raw_message_is_none_for_other_classes() -> None:
    """Sibling classes default ``raw_message`` to ``None`` (no
    server text in scope on a purely client-side raise)."""
    assert DqliteConnectionError("x").raw_message is None
    assert ProtocolError("x").raw_message is None
    assert InterfaceError("x").raw_message is None
    assert ClusterError("x").raw_message is None
    assert DataError("x").raw_message is None
