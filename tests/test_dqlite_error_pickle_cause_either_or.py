"""Pin: ``DqliteError.__cause__`` survives pickle either-or — the
restored exception's ``__cause__`` is either ``None`` (default
pickle behaviour drops chained exceptions when ``__getstate__``
does not capture them) or an instance of the original cause class.

Mirrors the wire-layer pin at
``python-dqlite-wire/tests/test_exceptions.py::test_poisoned_error_cause_pickle_behaviour``.
The deliberate-loss posture is acceptable today because the SA
dialect's ``is_disconnect`` reads ``raw_message`` and ``code``
first (both are preserved in client-side ``__getstate__``); the
class-chain is diagnostically lossy across cross-process pickling
but not classifier-breaking. Pinning the either-or shape forecloses
a future change that silently captures a non-picklable cause and
breaks cross-process exception transfer (``ProcessPoolExecutor``,
``multiprocessing.Queue``, Celery result backends).
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
    # Either default-pickle drops the cause OR the (still-picklable)
    # cause is preserved. Both are acceptable; what is NOT acceptable
    # is partial state (a non-None __cause__ that is the wrong type
    # or has lost its forensic state).
    if restored.__cause__ is not None:
        assert isinstance(restored.__cause__, ValueError)
        assert "inner forensic state" in str(restored.__cause__)
    # Leaf-level forensic state survives regardless of __cause__:
    assert restored.code == 42
    assert "outer wrap" in str(restored)


def test_dqlite_error_cause_deepcopy_either_or() -> None:
    """deepcopy goes through __reduce__ same as pickle — pin the
    same either-or contract."""
    inner = RuntimeError("inner")
    outer = DqliteError("outer")
    outer.__cause__ = inner
    restored = copy.deepcopy(outer)
    if restored.__cause__ is not None:
        assert isinstance(restored.__cause__, RuntimeError)


def test_operational_error_cause_pickle_either_or() -> None:
    """OperationalError carries an extra display-message cap and a
    distinct ``__init__`` signature; pin its cause-pickling contract
    too so the OE-arm of the dialect's classifier is not a blind
    spot."""
    inner = ValueError("wire-level decode failure")
    outer = OperationalError("server message", 1)
    outer.__cause__ = inner
    restored = pickle.loads(pickle.dumps(outer))
    if restored.__cause__ is not None:
        assert isinstance(restored.__cause__, ValueError)
    assert restored.code == 1
    assert restored.message == "server message"
