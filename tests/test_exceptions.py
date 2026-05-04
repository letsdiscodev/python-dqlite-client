"""Tests for the dqliteclient.exceptions module."""

from __future__ import annotations

import copy
import pickle

import pytest

from dqliteclient.exceptions import OperationalError


class TestOperationalErrorPickling:
    """``OperationalError`` must round-trip through pickle unchanged.

    SQLAlchemy's multiprocessing pool, Celery's result backend, and
    ``concurrent.futures.ProcessPoolExecutor`` all rely on
    ``Exception.__reduce_ex__`` to serialize exceptions across process
    boundaries. Previously the custom ``__init__`` stored a
    pre-formatted string in ``self.args`` (``"[code] message"``), so
    the unpickler called ``OperationalError("[code] message")`` — one
    positional argument for a two-parameter constructor — and raised
    ``TypeError``.
    """

    def test_pickle_roundtrip_preserves_fields(self) -> None:
        original = OperationalError(5, "boom")
        restored = pickle.loads(pickle.dumps(original))
        assert isinstance(restored, OperationalError)
        assert restored.code == 5
        assert restored.message == "boom"

    def test_pickle_roundtrip_preserves_str(self) -> None:
        original = OperationalError(19, "constraint failed")
        restored = pickle.loads(pickle.dumps(original))
        assert str(restored) == "[19] constraint failed"

    def test_deepcopy_preserves_fields(self) -> None:
        original = OperationalError(1555, "not null")
        clone = copy.deepcopy(original)
        assert clone.code == 1555
        assert clone.message == "not null"
        assert str(clone) == "[1555] not null"

    def test_copy_preserves_fields(self) -> None:
        original = OperationalError(1299, "check")
        clone = copy.copy(original)
        assert clone.code == 1299
        assert clone.message == "check"

    @pytest.mark.parametrize("code,message", [(0, ""), (-1, "neg"), (2**31, "high")])
    def test_pickle_parametrized(self, code: int, message: str) -> None:
        original = OperationalError(code, message)
        restored = pickle.loads(pickle.dumps(original))
        assert restored.code == code
        assert restored.message == message
        assert str(restored) == f"[{code}] {message}"


class TestOperationalErrorFormatting:
    """``str()`` keeps the ``[code] message`` format for log back-compat."""

    def test_str_format(self) -> None:
        e = OperationalError(5, "boom")
        assert str(e) == "[5] boom"

    def test_repr_contains_both_fields(self) -> None:
        e = OperationalError(5, "boom")
        r = repr(e)
        # Exact repr shape is not pinned; just ensure nothing is lost.
        assert "5" in r
        assert "boom" in r


class TestOperationalErrorMessageTruncation:
    """Large ``FailureResponse.message`` values must not inflate every
    traceback / log line. The wire caps at 64 KiB; ``OperationalError``
    truncates to ~1 KiB for display while keeping the full payload on
    ``.raw_message`` for forensic access.
    """

    def test_long_message_is_truncated_for_display(self) -> None:
        payload = "x" * 63000
        e = OperationalError(5, payload)
        assert len(e.message) < 1200, "display message must be truncated to avoid log amplification"
        assert "truncated" in e.message

    def test_raw_message_capped_at_4kb_for_payload_safety(self) -> None:
        """``raw_message`` is bounded to ~4 KiB so cross-process
        pickled exception graphs stay small even under hostile-peer
        fan-out. The wire layer caps a single FailureResponse at
        ~64 KiB; combined with BaseExceptionGroup chains, an
        unbounded raw_message produced multi-MB pickled payloads."""
        payload = "x" * 63000
        e = OperationalError(5, payload)
        # raw_message is truncated with a marker that exposes the
        # original size class for triage.
        assert len(e.raw_message) < 5000, "raw_message must be capped"
        assert "raw_message truncated" in e.raw_message

    def test_short_message_is_not_touched(self) -> None:
        e = OperationalError(5, "ordinary error")
        assert e.message == "ordinary error"
        assert e.raw_message == "ordinary error"
        assert "truncated" not in e.message

    def test_pickle_roundtrip_is_lossless_within_caps(self) -> None:
        """Pickling preserves raw_message and re-applies display
        truncation. With raw_message itself capped, large payloads
        round-trip with their bounded form intact."""
        payload = "y" * 5000
        original = OperationalError(19, payload)
        restored = pickle.loads(pickle.dumps(original))
        # raw_message survives the pickle round-trip (the cap was
        # applied at construction; the bounded value is what's
        # pickled).
        assert restored.raw_message == original.raw_message
        # After round-trip the display truncation is re-applied.
        assert len(restored.message) < 1200
        assert "truncated" in restored.message
