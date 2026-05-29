"""Pin: ``LeaderInfo`` / ``NodeMetadata`` are hashable, repr/pickle/deepcopy-stable value
types. Dropping ``frozen=True`` breaks hashing while ``__eq__`` keeps working — hidden."""

from __future__ import annotations

import copy
import pickle

from dqliteclient import LeaderInfo, NodeMetadata


def test_leader_info_hashable_and_set_keyable() -> None:
    a = LeaderInfo(node_id=1, address="n1:9001")
    b = LeaderInfo(node_id=1, address="n1:9001")
    c = LeaderInfo(node_id=2, address="n2:9001")
    assert hash(a) == hash(b)
    assert {a, c, b} == {a, c}


def test_leader_info_pickle_round_trip() -> None:
    li = LeaderInfo(node_id=3, address="host:9999")
    restored = pickle.loads(pickle.dumps(li))
    assert restored == li
    assert hash(restored) == hash(li)


def test_leader_info_deepcopy_equal() -> None:
    li = LeaderInfo(node_id=4, address="x:1")
    assert copy.deepcopy(li) == li


def test_leader_info_repr_includes_fields() -> None:
    li = LeaderInfo(node_id=42, address="host:9999")
    rendered = repr(li)
    assert "42" in rendered
    assert "host:9999" in rendered


def test_node_metadata_hashable_and_set_keyable() -> None:
    a = NodeMetadata(failure_domain=1, weight=5)
    b = NodeMetadata(failure_domain=1, weight=5)
    c = NodeMetadata(failure_domain=2, weight=5)
    assert hash(a) == hash(b)
    assert {a, c, b} == {a, c}


def test_node_metadata_pickle_round_trip() -> None:
    nm = NodeMetadata(failure_domain=42, weight=7)
    restored = pickle.loads(pickle.dumps(nm))
    assert restored == nm
    assert hash(restored) == hash(nm)


def test_node_metadata_deepcopy_equal() -> None:
    nm = NodeMetadata(failure_domain=99, weight=1)
    assert copy.deepcopy(nm) == nm


def test_node_metadata_repr_includes_fields() -> None:
    nm = NodeMetadata(failure_domain=77, weight=3)
    rendered = repr(nm)
    assert "77" in rendered
    assert "3" in rendered
