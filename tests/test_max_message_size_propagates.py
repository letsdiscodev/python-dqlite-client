"""Pin: the wire-layer ``max_message_size`` knob is propagated to every client entry point
so operators can loosen it for large dumps or tighten it as defense in depth."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

import dqliteclient
from dqliteclient.connection import DqliteConnection
from dqliteclient.pool import ConnectionPool
from dqliteclient.protocol import DEFAULT_MAX_MESSAGE_SIZE, DqliteProtocol


def test_default_max_message_size_re_exported_from_wire_layer() -> None:
    """Client ``DEFAULT_MAX_MESSAGE_SIZE`` tracks the wire constant, no code change on bump."""
    from dqlitewire import ReadBuffer

    assert DEFAULT_MAX_MESSAGE_SIZE == ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE
    assert DEFAULT_MAX_MESSAGE_SIZE == 64 * 1024 * 1024


def test_default_max_message_size_in_package_top_level_all() -> None:
    assert "DEFAULT_MAX_MESSAGE_SIZE" in dqliteclient.__all__
    assert dqliteclient.DEFAULT_MAX_MESSAGE_SIZE is DEFAULT_MAX_MESSAGE_SIZE


def test_max_message_size_kwarg_on_every_entry_point() -> None:
    """Every public entry point accepts ``max_message_size`` (default None = wire default)."""
    for fn in (
        dqliteclient.connect,
        dqliteclient.create_pool,
        DqliteConnection.__init__,
        ConnectionPool.__init__,
    ):
        sig = inspect.signature(fn)
        assert "max_message_size" in sig.parameters, (
            f"{fn!r} is missing the ``max_message_size`` parameter"
        )
        param = sig.parameters["max_message_size"]
        assert param.default is None, (
            f"{fn!r}'s ``max_message_size`` default must be None "
            f"(meaning 'wire-layer default'), got {param.default!r}"
        )


def test_max_message_size_kwarg_on_cluster_connect() -> None:
    from dqliteclient.cluster import ClusterClient

    sig = inspect.signature(ClusterClient.connect)
    assert "max_message_size" in sig.parameters
    assert sig.parameters["max_message_size"].default is None


def test_max_message_size_reaches_decoder_via_dqlite_protocol() -> None:
    """A tightened ``max_message_size`` reaches the inner ``MessageDecoder`` buffer cap."""
    reader = MagicMock()
    writer = MagicMock()
    proto = DqliteProtocol(reader, writer, max_message_size=1024)
    assert proto._decoder._max_message_size == 1024
    assert proto._decoder._buffer._max_message_size == 1024


def test_max_message_size_none_defers_to_wire_default() -> None:
    """``None`` selects ``DEFAULT_MAX_MESSAGE_SIZE``."""
    reader = MagicMock()
    writer = MagicMock()
    proto = DqliteProtocol(reader, writer)
    assert proto._decoder._max_message_size == DEFAULT_MAX_MESSAGE_SIZE


def test_max_message_size_invalid_value_rejected_at_protocol_layer() -> None:
    """Invalid values are rejected at construction, not deep inside ``ReadBuffer``."""
    reader = MagicMock()
    writer = MagicMock()
    with pytest.raises(TypeError, match="max_message_size must be int or None"):
        DqliteProtocol(reader, writer, max_message_size="64")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_message_size must be >= 1"):
        DqliteProtocol(reader, writer, max_message_size=0)


def test_max_message_size_threaded_through_dqlite_connection() -> None:
    """``DqliteConnection`` stores ``max_message_size`` for forwarding at ``connect()``."""
    conn = DqliteConnection("localhost:9001", max_message_size=8192)
    assert conn._max_message_size == 8192

    conn_default = DqliteConnection("localhost:9001")
    assert conn_default._max_message_size is None


def test_max_message_size_threaded_through_connection_pool() -> None:
    """``ConnectionPool`` stores ``max_message_size`` for every pooled connection."""
    pool = ConnectionPool(["localhost:9001"], max_message_size=2048)
    assert pool._max_message_size == 2048
