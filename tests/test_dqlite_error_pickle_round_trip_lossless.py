"""Pin: every ``DqliteError`` subclass round-trips through pickle /
deepcopy without losing the ``raw_message`` (base) or ``code``
(``DqliteConnectionError``) fields.

The fields were added in cycle 27 (XP2 / XP3) so the wire-level
signal would survive cross-process boundaries — ``ProcessPoolExecutor``,
``multiprocessing.Queue.put(exception)``, Celery task results, SA's
multiprocess pool. Without overriding ``__reduce__`` the default
``Exception`` pickle path (``(cls, self.args)``) silently drops every
attribute set on the instance after ``Exception.__init__``.

Pin lossless round-trip on every subclass that takes ``raw_message=``
or ``code=`` so a future regression on the ``__reduce__`` discipline
fails this test.
"""

from __future__ import annotations

import copy
import pickle

import pytest

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
    """No-arg / message-only constructions still round-trip cleanly."""
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
    """Defence pin: OperationalError pickle preserves code and the
    bounded raw_message (the cap is applied at construction; the
    bounded value survives the round-trip). Display message is
    re-truncated."""
    payload = "y" * 5000
    e = OperationalError(19, payload)
    restored = pickle.loads(pickle.dumps(e))
    # raw_message round-trips with whatever bounded value was set
    # at construction.
    assert restored.raw_message == e.raw_message
    assert restored.code == 19
    assert len(restored.message) < 1200
    assert "truncated" in restored.message


def test_pickle_round_trip_through_multiprocessing_queue() -> None:
    """End-to-end: an exception sent through ``multiprocessing.Queue``
    survives with code and raw_message intact. This is the canonical
    cross-process surface XP2 was added to plumb."""
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
