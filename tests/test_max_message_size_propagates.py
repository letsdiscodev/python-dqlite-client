"""Pin: the wire-layer ``max_message_size`` knob is propagated as a
constructor parameter on every client entry point — ``DqliteProtocol``,
``DqliteConnection``, ``ConnectionPool``, ``ClusterClient.connect``,
``dqliteclient.connect``, ``dqliteclient.create_pool``.

Before this propagation, the 64 MiB wire-layer default
(``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE``) was the only value any client
caller could ever see. Operators running large-Dump backups could not
loosen the cap; operators on small-payload workloads could not tighten
it as defense in depth. Sibling governors (``max_total_rows`` /
``max_continuation_frames``) were already propagated; this is the third
wire governor finally aligned with the same pattern.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

import dqliteclient
from dqliteclient.connection import DqliteConnection
from dqliteclient.pool import ConnectionPool
from dqliteclient.protocol import DEFAULT_MAX_MESSAGE_SIZE, DqliteProtocol


def test_default_max_message_size_re_exported_from_wire_layer() -> None:
    """The client-layer ``DEFAULT_MAX_MESSAGE_SIZE`` constant matches
    the wire-layer's ``ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE`` so a wire
    bump propagates without a code change here."""
    from dqlitewire import ReadBuffer

    assert DEFAULT_MAX_MESSAGE_SIZE == ReadBuffer.DEFAULT_MAX_MESSAGE_SIZE
    assert DEFAULT_MAX_MESSAGE_SIZE == 64 * 1024 * 1024


def test_default_max_message_size_in_package_top_level_all() -> None:
    assert "DEFAULT_MAX_MESSAGE_SIZE" in dqliteclient.__all__
    assert dqliteclient.DEFAULT_MAX_MESSAGE_SIZE is DEFAULT_MAX_MESSAGE_SIZE


def test_max_message_size_kwarg_on_every_entry_point() -> None:
    """Every public entry point must accept ``max_message_size`` as a
    keyword parameter (default ``None``, meaning "use the wire-layer
    default"). Mirrors the sibling pattern for ``max_total_rows`` /
    ``max_continuation_frames``."""
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
    """Construct a ``DqliteProtocol`` with a tightened
    ``max_message_size`` and assert the inner ``MessageDecoder``'s
    buffer cap reflects the override."""
    reader = MagicMock()
    writer = MagicMock()
    proto = DqliteProtocol(reader, writer, max_message_size=1024)
    # The wire-layer ReadBuffer stores the cap as
    # ``_max_message_size``; the codec promotes it to
    # ``MessageDecoder._max_message_size`` too.
    assert proto._decoder._max_message_size == 1024
    assert proto._decoder._buffer._max_message_size == 1024


def test_max_message_size_none_defers_to_wire_default() -> None:
    """Passing ``None`` (the default) selects
    ``DEFAULT_MAX_MESSAGE_SIZE`` (the wire layer's 64 MiB)."""
    reader = MagicMock()
    writer = MagicMock()
    proto = DqliteProtocol(reader, writer)
    assert proto._decoder._max_message_size == DEFAULT_MAX_MESSAGE_SIZE


def test_max_message_size_invalid_value_rejected_at_protocol_layer() -> None:
    """The client-layer validation rejects non-int / non-positive
    values up front so a misconfigured DSN surfaces ``TypeError`` /
    ``ValueError`` at construction, not deep inside ``ReadBuffer``."""
    reader = MagicMock()
    writer = MagicMock()
    with pytest.raises(TypeError, match="max_message_size must be int or None"):
        DqliteProtocol(reader, writer, max_message_size="64")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_message_size must be >= 1"):
        DqliteProtocol(reader, writer, max_message_size=0)


def test_max_message_size_threaded_through_dqlite_connection() -> None:
    """``DqliteConnection`` accepts ``max_message_size`` and stores it
    for later forwarding to the ``DqliteProtocol`` it constructs on
    ``connect()``."""
    conn = DqliteConnection("localhost:9001", max_message_size=8192)
    assert conn._max_message_size == 8192

    # Default None is stored verbatim; the wire-layer default kicks in
    # only at protocol-construction time.
    conn_default = DqliteConnection("localhost:9001")
    assert conn_default._max_message_size is None


def test_max_message_size_threaded_through_connection_pool() -> None:
    """``ConnectionPool`` accepts ``max_message_size`` and stores it
    for forwarding to every pooled connection."""
    pool = ConnectionPool(["localhost:9001"], max_message_size=2048)
    assert pool._max_message_size == 2048
