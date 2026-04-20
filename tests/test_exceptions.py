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
