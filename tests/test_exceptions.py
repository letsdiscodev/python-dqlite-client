"""Exception hierarchy: raw messages, leader codes, repr/str shape, sanitisation and pickling."""

from __future__ import annotations

import copy
import pickle

import pytest

import dqliteclient.exceptions as ce
from dqliteclient.connection import DqliteConnection
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


class TestOperationalErrorPickling:
    """OperationalError round-trips through pickle unchanged (the two-arg __init__ once broke
    the default unpickle path that replayed self.args positionally)."""

    def test_pickle_roundtrip_preserves_fields(self) -> None:
        original = OperationalError("boom", 5)
        restored = pickle.loads(pickle.dumps(original))
        assert isinstance(restored, OperationalError)
        assert restored.code == 5
        assert restored.message == "boom"

    def test_pickle_roundtrip_preserves_str(self) -> None:
        original = OperationalError("constraint failed", 19)
        restored = pickle.loads(pickle.dumps(original))
        assert str(restored) == "constraint failed"

    def test_deepcopy_preserves_fields(self) -> None:
        original = OperationalError("not null", 1555)
        clone = copy.deepcopy(original)
        assert clone.code == 1555
        assert clone.message == "not null"
        assert str(clone) == "not null"

    def test_copy_preserves_fields(self) -> None:
        original = OperationalError("check", 1299)
        clone = copy.copy(original)
        assert clone.code == 1299
        assert clone.message == "check"

    @pytest.mark.parametrize("code,message", [(0, ""), (-1, "neg"), (2**31, "high")])
    def test_pickle_parametrized(self, code: int, message: str) -> None:
        original = OperationalError(message, code)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.code == code
        assert restored.message == message
        assert str(restored) == message


class TestOperationalErrorFormatting:
    """str() returns the bare message (matching sqlite3.OperationalError); code stays on .code."""

    def test_str_format(self) -> None:
        e = OperationalError("boom", 5)
        assert str(e) == "boom"

    def test_repr_contains_both_fields(self) -> None:
        e = OperationalError("boom", 5)
        r = repr(e)
        assert "5" in r
        assert "boom" in r


class TestOperationalErrorMessageTruncation:
    """OperationalError truncates message to ~1 KiB for display, keeping the payload on
    .raw_message, so large FailureResponse values don't inflate every traceback/log line."""

    def test_long_message_is_truncated_for_display(self) -> None:
        payload = "x" * 63000
        e = OperationalError(payload, 5)
        assert len(e.message) < 1200, "display message must be truncated to avoid log amplification"
        assert "truncated" in e.message

    def test_raw_message_capped_at_4kb_for_payload_safety(self) -> None:
        """raw_message is capped at ~4 KiB so hostile-peer 64 KiB payloads in BaseExceptionGroup
        chains can't balloon cross-process pickled exception graphs to multi-MB."""
        payload = "x" * 63000
        e = OperationalError(payload, 5)
        assert len(e.raw_message) < 5000, "raw_message must be capped"
        assert "raw_message truncated" in e.raw_message

    def test_short_message_is_not_touched(self) -> None:
        e = OperationalError("ordinary error", 5)
        assert e.message == "ordinary error"
        assert e.raw_message == "ordinary error"
        assert "truncated" not in e.message

    def test_pickle_roundtrip_is_lossless_within_caps(self) -> None:
        """Pickling preserves the bounded raw_message and re-applies display truncation."""
        payload = "y" * 5000
        original = OperationalError(payload, 19)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.raw_message == original.raw_message
        assert len(restored.message) < 1200
        assert "truncated" in restored.message


class TestDqliteErrorRawMessageCap:
    """The ~4 KiB raw_message cap lives on DqliteError so every code-bearing subclass inherits
    it, closing every path a hostile-peer payload could flow uncapped into a pickle graph."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (DqliteConnectionError, {"code": 10250}),
            (InterfaceError, {}),
            (ClusterError, {}),
            (DataError, {}),
        ],
    )
    def test_raw_message_capped_on_dqlite_error_subclass(
        self, cls: type, kwargs: dict[str, object]
    ) -> None:
        big = "X" * 63_000
        e = cls("trunc msg", raw_message=big, **kwargs)
        assert e.raw_message is not None
        assert len(e.raw_message) < 5000
        assert "raw_message truncated" in e.raw_message

    def test_short_raw_message_is_not_touched(self) -> None:
        short = "ordinary error"
        e = DqliteConnectionError("msg", code=1, raw_message=short)
        assert e.raw_message == short

    def test_none_raw_message_round_trips(self) -> None:
        e = DqliteConnectionError("msg", code=1, raw_message=None)
        assert e.raw_message is None


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


def test_dqlite_connection_error_cause_pickle_either_or() -> None:
    inner = ValueError("inner forensic state")
    outer = DqliteConnectionError("outer wrap", code=42)
    outer.__cause__ = inner
    restored = pickle.loads(pickle.dumps(outer))
    # Either the cause is dropped or preserved intact — never partial state.
    if restored.__cause__ is not None:
        assert isinstance(restored.__cause__, ValueError)
        assert "inner forensic state" in str(restored.__cause__)
    assert restored.code == 42
    assert "outer wrap" in str(restored)


def test_dqlite_error_cause_deepcopy_either_or() -> None:
    """deepcopy goes through __reduce__ like pickle: same either-or contract."""
    inner = RuntimeError("inner")
    outer = DqliteError("outer")
    outer.__cause__ = inner
    restored = copy.deepcopy(outer)
    if restored.__cause__ is not None:
        assert isinstance(restored.__cause__, RuntimeError)


def test_operational_error_cause_pickle_either_or() -> None:
    """OperationalError has a distinct __init__ signature; pin its
    cause-pickling contract too."""
    inner = ValueError("wire-level decode failure")
    outer = OperationalError("server message", 1)
    outer.__cause__ = inner
    restored = pickle.loads(pickle.dumps(outer))
    if restored.__cause__ is not None:
        assert isinstance(restored.__cause__, ValueError)
    assert restored.code == 1
    assert restored.message == "server message"


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_dqlite_connection_error_pickle_preserves_code(protocol: int) -> None:
    e = DqliteConnectionError(
        "Node host:9001 is no longer leader: not leader",
        code=10250,
        raw_message="not leader",
    )
    restored = pickle.loads(pickle.dumps(e, protocol=protocol))
    assert restored.code == 10250
    assert restored.raw_message == "not leader"
    assert "Node host:9001" in str(restored)


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_dqlite_connection_error_deepcopy_preserves_code(protocol: int) -> None:
    e = DqliteConnectionError("leader-flip", code=10506, raw_message="leadership lost")
    restored = copy.deepcopy(e)
    assert restored.code == 10506
    assert restored.raw_message == "leadership lost"


def test_dqlite_connection_error_default_construction_pickle_round_trip() -> None:
    e = DqliteConnectionError("Connection refused")
    restored = pickle.loads(pickle.dumps(e))
    assert restored.code is None
    assert restored.raw_message is None
    assert str(restored) == "Connection refused"


@pytest.mark.parametrize(
    "cls",
    [DqliteError, DataError, InterfaceError, ClusterError, ClusterPolicyError, ProtocolError],
)
def test_subclass_raw_message_round_trips_through_pickle(cls: type) -> None:
    e = cls("msg", raw_message="server text")
    restored = pickle.loads(pickle.dumps(e))
    assert restored.raw_message == "server text"
    assert str(restored) == "msg"


@pytest.mark.parametrize(
    "cls",
    [DqliteError, DataError, InterfaceError, ClusterError, ClusterPolicyError, ProtocolError],
)
def test_subclass_raw_message_round_trips_through_deepcopy(cls: type) -> None:
    e = cls("msg", raw_message="server text")
    restored = copy.deepcopy(e)
    assert restored.raw_message == "server text"


def test_operational_error_pickle_lossless_within_caps() -> None:
    """OperationalError pickle preserves code and bounded raw_message; display re-truncated."""
    payload = "y" * 5000
    e = OperationalError(payload, 19)
    restored = pickle.loads(pickle.dumps(e))
    assert restored.raw_message == e.raw_message
    assert restored.code == 19
    assert len(restored.message) < 1200
    assert "truncated" in restored.message


def test_pickle_round_trip_through_multiprocessing_queue() -> None:
    """An exception sent through multiprocessing.Queue survives with code/raw_message intact."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    e = DqliteConnectionError(
        "node failover mid-handshake",
        code=10250,
        raw_message="not leader",
    )
    q.put(e)
    restored = q.get(timeout=5)
    assert isinstance(restored, DqliteConnectionError)
    assert restored.code == 10250
    assert restored.raw_message == "not leader"


def test_dqlite_connection_error_default_construction_works() -> None:
    """Backwards-compat: positional message gives code=None / raw_message=None."""
    e = DqliteConnectionError("Connection refused")
    assert str(e) == "Connection refused"
    assert e.code is None
    assert e.raw_message is None


def test_dqlite_connection_error_carries_code_and_raw_message() -> None:
    e = DqliteConnectionError(
        "Node leader-a:9001 is no longer leader: not leader",
        code=10250,
        raw_message="not leader",
    )
    assert e.code == 10250
    assert e.raw_message == "not leader"


def test_dqlite_connection_error_is_dqlite_error_subclass() -> None:
    assert issubclass(DqliteConnectionError, DqliteError)


def test_no_args_construction_works() -> None:
    e = DqliteConnectionError()
    assert e.code is None
    assert e.raw_message is None


def test_dqlite_connection_error_repr_includes_code_when_set() -> None:
    e = DqliteConnectionError("Not leader", code=10250, raw_message="not leader")
    r = repr(e)
    assert "10250" in r, (
        f"DqliteConnectionError.__repr__ must include code=N when set; "
        f"got {r!r}. Mirrors the dbapi DatabaseError.__repr__ discipline."
    )
    assert "DqliteConnectionError" in r
    assert "'Not leader'" in r or '"Not leader"' in r


def test_dqlite_connection_error_repr_omits_code_when_none() -> None:
    """Negative twin: code=None renders without the noisy code= suffix."""
    e = DqliteConnectionError("Generic transport fault")
    r = repr(e)
    assert "DqliteConnectionError" in r
    assert "code=" not in r, (
        f"DqliteConnectionError.__repr__ must omit code= when code is None; got {r!r}"
    )


def test_dqlite_connection_error_repr_handles_empty_message() -> None:
    """A no-args DqliteConnectionError still reprs cleanly (message defaults to "")."""
    e = DqliteConnectionError(code=10250)
    r = repr(e)
    assert "DqliteConnectionError" in r
    assert "10250" in r


def test_dqlite_connection_repr_includes_id() -> None:
    a = DqliteConnection("127.0.0.1:9001", database="db")
    b = DqliteConnection("127.0.0.1:9001", database="db")

    repr_a = repr(a)
    repr_b = repr(b)

    assert " at 0x" in repr_a, repr_a
    assert " at 0x" in repr_b, repr_b
    assert hex(id(a))[2:] in repr_a
    assert hex(id(b))[2:] in repr_b
    assert repr_a != repr_b


def test_dqlite_connection_repr_state_changes_visible() -> None:
    conn = DqliteConnection("127.0.0.1:9001", database="db")

    r = repr(conn)
    assert "disconnected" in r
    assert "127.0.0.1:9001" in r
    assert " at 0x" in r


def test_operational_error_message_strips_cr() -> None:
    """Scrubs CR (and most control bytes) but intentionally preserves LF and Tab."""
    e = OperationalError("hello\rWARNING: faked log line", 1)
    assert "\r" not in str(e)


def test_operational_error_message_strips_nul() -> None:
    e = OperationalError("hello\x00world", 1)
    assert "\x00" not in str(e)


def test_operational_error_message_strips_ansi_escape() -> None:
    e = OperationalError("\x1b[31mred\x1b[0m", 1)
    assert "\x1b" not in str(e)


def test_operational_error_raw_message_preserves_unsanitised_text() -> None:
    """``raw_message`` stays verbatim; only the display field is sanitised."""
    raw = "hello\r\nWARNING\x00\x1b[31m"
    e = OperationalError("display", 1, raw_message=raw)
    assert e.raw_message == raw


def test_operational_error_raw_message_defaults_to_unsanitised_message() -> None:
    """When omitted, ``raw_message`` is populated from ``message`` BEFORE sanitisation."""
    raw = "hello\r\nWARNING\x00"
    e = OperationalError(raw, 1)
    assert e.raw_message == raw


def test_operational_error_message_idempotent_pre_sanitised() -> None:
    """Re-sanitising already-clean input is idempotent."""
    clean = "ordinary diagnostic"
    e = OperationalError(clean, 1)
    assert str(e) == clean


@pytest.mark.parametrize("ctrl", ["\x07", "\x0c", "\x1f"])
def test_operational_error_message_strips_other_control_bytes(ctrl: str) -> None:
    e = OperationalError(f"prefix{ctrl}suffix", 1)
    assert ctrl not in str(e)


def test_operational_error_repr_includes_labelled_code() -> None:
    err = OperationalError("simulated failure", 5)
    rendered = repr(err)
    assert "OperationalError(" in rendered
    assert "'simulated failure'" in rendered
    assert "code=5" in rendered


def test_operational_error_repr_quotes_message_via_repr() -> None:
    """Messages with embedded escape chars must be repr'd, not rendered literally."""
    err = OperationalError("line1\nline2", 12)
    rendered = repr(err)
    assert "\n" not in rendered
    assert "\\n" in rendered
    assert "code=12" in rendered


def test_operational_error_repr_truncated_message_renders_truncated_form() -> None:
    """repr uses the truncated display ``message``, not ``raw_message``, so it stays bounded."""
    long_msg = "a" * 2000
    err = OperationalError(long_msg, 99)
    rendered = repr(err)
    assert "[truncated" in rendered
    assert "code=99" in rendered


def test_client_operationalerror_positional_args_match_message_code() -> None:
    """Message first, code second, mirroring stdlib sqlite3 and dqlitedbapi."""
    e = ce.OperationalError("boom", 42)
    assert e.message == "boom"
    assert e.code == 42


def test_client_operationalerror_with_raw_message_kwarg() -> None:
    e = ce.OperationalError("display", 5, raw_message="full server text")
    assert e.message == "display"
    assert e.code == 5
    assert e.raw_message == "full server text"


def test_pickle_round_trip_preserves_positional_shape() -> None:
    """``__reduce__`` relies on ``self.args == (message, code)`` for round-trip."""
    import pickle

    e = ce.OperationalError("constraint failed", 19, raw_message="full text")
    restored = pickle.loads(pickle.dumps(e))
    assert restored.message == "constraint failed"
    assert restored.code == 19
    assert restored.raw_message == "full text"
