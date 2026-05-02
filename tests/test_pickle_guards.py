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

Mirror the ISSUE-791 pattern: every class raises a
driver-level ``TypeError`` naming the specific class.
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
