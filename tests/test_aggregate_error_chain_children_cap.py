"""Aggregate BaseExceptionGroup chains are bounded at _MAX_AGGREGATE_CHILDREN.

Excess children collapse into a synthetic DqliteError ("... and N more") to avoid a
multi-megabyte exception graph that survives cross-process pickling.
"""

import pickle

from dqliteclient.cluster import _MAX_AGGREGATE_CHILDREN, _bounded_group
from dqliteclient.exceptions import DqliteConnectionError, DqliteError


def test_bounded_group_passes_short_chain_unchanged() -> None:
    excs: list[BaseException] = [DqliteConnectionError(f"err {i}") for i in range(5)]
    group = _bounded_group("test", excs)
    assert len(group.exceptions) == 5
    assert all(isinstance(e, DqliteConnectionError) for e in group.exceptions)


def test_bounded_group_caps_long_chain() -> None:
    excs: list[BaseException] = [
        DqliteConnectionError(f"err {i}") for i in range(_MAX_AGGREGATE_CHILDREN + 10)
    ]
    group = _bounded_group("test", excs)
    assert len(group.exceptions) == _MAX_AGGREGATE_CHILDREN + 1
    overflow = group.exceptions[-1]
    assert isinstance(overflow, DqliteError)
    assert "10 more" in str(overflow)


def test_bounded_group_picklable() -> None:
    """The chain must round-trip through pickle for cross-process error capture."""
    excs: list[BaseException] = [DqliteConnectionError(f"err {i}") for i in range(50)]
    group = _bounded_group("test", excs)

    blob = pickle.dumps(group)
    restored = pickle.loads(blob)

    assert isinstance(restored, BaseExceptionGroup)
    assert len(restored.exceptions) == _MAX_AGGREGATE_CHILDREN + 1
    assert "30 more" in str(restored.exceptions[-1])


def test_bounded_group_at_exact_cap_no_overflow() -> None:
    """Boundary: exactly _MAX_AGGREGATE_CHILDREN passes through with no overflow synthetic."""
    excs: list[BaseException] = [
        DqliteConnectionError(f"err {i}") for i in range(_MAX_AGGREGATE_CHILDREN)
    ]
    group = _bounded_group("test", excs)
    assert len(group.exceptions) == _MAX_AGGREGATE_CHILDREN
    assert all(isinstance(e, DqliteConnectionError) for e in group.exceptions)
