"""Client-layer classes raise a class-named ``TypeError`` on pickle / copy /
deepcopy. Needed because asyncio primitives pickle silently on 3.10+, yielding
a "live"-looking duplicate with severed loop bindings."""

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
        """The class is lock-free (mutual exclusion via ``_in_use``), so the
        guard message must not name a phantom ``_op_lock``."""
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
    """Copying would build a fresh ``Lock``, breaking ``set_nodes``
    mutual-exclusion; guard against it."""

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
