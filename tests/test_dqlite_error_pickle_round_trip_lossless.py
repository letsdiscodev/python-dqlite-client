"""DqliteError subclasses round-trip raw_message/code through pickle and deepcopy.

The default Exception pickle path ``(cls, self.args)`` drops attributes set after
``Exception.__init__``; the subclasses override ``__reduce__`` to preserve them.
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
