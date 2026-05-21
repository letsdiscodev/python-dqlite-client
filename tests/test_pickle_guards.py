"""Pin: client-layer classes raise a clear ``TypeError`` on pickle /
copy / deepcopy. Symmetric with the dbapi's existing pickle guards
on Connection / Cursor.

Without explicit ``__reduce__`` raises:
- ``ConnectionPool`` / ``ClusterClient`` SILENTLY pickle (asyncio
  primitives became pickleable in 3.10+) — producing a
  "live"-looking duplicate detached from any loop. Any use yields
  opaque corruption.
- ``DqliteConnection`` (post- or pre-``connect()``) silently
  produces a corrupt duplicate via the default
  ``Exception.__reduce__`` walk; without an explicit reject the
  duplicate looks "live" but holds severed loop bindings and
  half-state.
- ``DqliteProtocol`` raises an opaque error from wrapped
  StreamReader / StreamWriter.

Each class raises a driver-level ``TypeError`` naming the specific
class, so the operator gets a precise diagnostic instead of a
generic pickling failure.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.connection import DqliteConnection
from dqliteclient.node_store import MemoryNodeStore
from dqliteclient.pool import ConnectionPool


class TestDqliteConnectionPickleGuard:
    def test_pickle_raises(self) -> None:
        conn = DqliteConnection("localhost:9001")
        with pytest.raises(TypeError, match="DqliteConnection"):
            pickle.dumps(conn)

    def test_copy_copy_raises(self) -> None:
        conn = DqliteConnection("localhost:9001")
        with pytest.raises(TypeError, match="DqliteConnection"):
            copy.copy(conn)

    def test_copy_deepcopy_raises(self) -> None:
        conn = DqliteConnection("localhost:9001")
        with pytest.raises(TypeError, match="DqliteConnection"):
            copy.deepcopy(conn)

    def test_message_does_not_reference_nonexistent_op_lock(self) -> None:
        """The class is intentionally lock-free (mutual exclusion is via
        the ``_in_use: bool`` flag, no ``asyncio.Lock`` instance). The
        pickle-guard message must not name an ``_op_lock`` field that
        does not exist on the class — operators following that hint
        would chase a phantom attribute."""
        conn = DqliteConnection("localhost:9001")
        assert not hasattr(conn, "_op_lock")
        with pytest.raises(TypeError) as ei:
            pickle.dumps(conn)
        assert "_op_lock" not in str(ei.value)
        assert "asyncio.Lock" not in str(ei.value)


class TestConnectionPoolPickleGuard:
    def test_pickle_raises(self) -> None:
        pool = ConnectionPool(addresses=["localhost:9001"])
        with pytest.raises(TypeError, match="ConnectionPool"):
            pickle.dumps(pool)

    def test_copy_copy_raises(self) -> None:
        pool = ConnectionPool(addresses=["localhost:9001"])
        with pytest.raises(TypeError, match="ConnectionPool"):
            copy.copy(pool)

    def test_copy_deepcopy_raises(self) -> None:
        pool = ConnectionPool(addresses=["localhost:9001"])
        with pytest.raises(TypeError, match="ConnectionPool"):
            copy.deepcopy(pool)


class TestClusterClientPickleGuard:
    def test_pickle_raises(self) -> None:
        cluster = ClusterClient(node_store=MemoryNodeStore(["localhost:9001"]))
        with pytest.raises(TypeError, match="ClusterClient"):
            pickle.dumps(cluster)

    def test_copy_copy_raises(self) -> None:
        cluster = ClusterClient(node_store=MemoryNodeStore(["localhost:9001"]))
        with pytest.raises(TypeError, match="ClusterClient"):
            copy.copy(cluster)

    def test_copy_deepcopy_raises(self) -> None:
        cluster = ClusterClient(node_store=MemoryNodeStore(["localhost:9001"]))
        with pytest.raises(TypeError, match="ClusterClient"):
            copy.deepcopy(cluster)


class TestMemoryNodeStorePickleGuard:
    """``MemoryNodeStore`` holds an eager ``asyncio.Lock`` for the
    ``set_nodes`` mutual-exclusion contract. Pickle / copy / deepcopy
    would construct a FRESH ``Lock`` on the copy, so the original
    and the duplicate would each pass through their own lock without
    serialisation — the documented single-owner-per-store discipline
    silently breaks. Symmetric with the seven sibling guards.
    """

    def test_pickle_raises(self) -> None:
        ns = MemoryNodeStore(addresses=["a.example:9001"])
        with pytest.raises(TypeError, match="MemoryNodeStore"):
            pickle.dumps(ns)

    def test_copy_copy_raises(self) -> None:
        ns = MemoryNodeStore(addresses=["a.example:9001"])
        with pytest.raises(TypeError, match="MemoryNodeStore"):
            copy.copy(ns)

    def test_copy_deepcopy_raises(self) -> None:
        ns = MemoryNodeStore(addresses=["a.example:9001"])
        with pytest.raises(TypeError, match="MemoryNodeStore"):
            copy.deepcopy(ns)


class TestDqliteProtocolPickleGuard:
    def test_pickle_raises(self) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from dqliteclient.protocol import DqliteProtocol

        reader = MagicMock(spec=asyncio.StreamReader)
        writer = MagicMock(spec=asyncio.StreamWriter)
        proto = DqliteProtocol(reader=reader, writer=writer)
        with pytest.raises(TypeError, match="DqliteProtocol"):
            pickle.dumps(proto)

    def test_copy_copy_raises(self) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from dqliteclient.protocol import DqliteProtocol

        reader = MagicMock(spec=asyncio.StreamReader)
        writer = MagicMock(spec=asyncio.StreamWriter)
        proto = DqliteProtocol(reader=reader, writer=writer)
        with pytest.raises(TypeError, match="DqliteProtocol"):
            copy.copy(proto)

    def test_copy_deepcopy_raises(self) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from dqliteclient.protocol import DqliteProtocol

        reader = MagicMock(spec=asyncio.StreamReader)
        writer = MagicMock(spec=asyncio.StreamWriter)
        proto = DqliteProtocol(reader=reader, writer=writer)
        with pytest.raises(TypeError, match="DqliteProtocol"):
            copy.deepcopy(proto)
