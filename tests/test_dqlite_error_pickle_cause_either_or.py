"""Pin: ``DqliteError.__cause__`` survives pickle either-or — restored
``__cause__`` is either ``None`` or an instance of the original cause
class, never partial state. Forecloses a future change that captures a
non-picklable cause and breaks cross-process exception transfer.
"""

from __future__ import annotations

import copy
import pickle

from dqliteclient.exceptions import (
    DqliteConnectionError,
    DqliteError,
    OperationalError,
)


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
