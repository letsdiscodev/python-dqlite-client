from __future__ import annotations

import copy
import pickle

import pytest

from dqliteclient.exceptions import (
    ClusterError,
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
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
