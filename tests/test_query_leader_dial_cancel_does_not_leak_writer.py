"""Pin: ``ClusterClient._query_leader`` / ``open_admin_connection`` and
``DqliteConnection._connect_impl`` use ``async with asyncio.timeout``
(not ``wait_for(open_connection(...))``, which discards the inner
``(reader, writer)`` on outer-cancel and leaks the socket + reader Task).
Source-level pins assert the leaky composite is gone.
"""

from __future__ import annotations

import inspect

from dqliteclient import cluster as cluster_module
from dqliteclient import connection as connection_module


def _source_of(name: str) -> str:
    obj = getattr(cluster_module.ClusterClient, name)
    return inspect.getsource(obj)


def _connection_source_of(name: str) -> str:
    obj = getattr(connection_module.DqliteConnection, name)
    return inspect.getsource(obj)


def test_query_leader_no_longer_uses_wait_for_around_open_connection() -> None:
    """``_query_leader`` uses ``asyncio.timeout`` for the dial."""
    src = _source_of("_query_leader")
    # Strip whitespace from both source and literal so the ``not in``
    # is not vacuously satisfied by differing indentation.
    src_stripped = src.replace(" ", "").replace("\n", "")
    assert "wait_for(open_connection(" not in src_stripped, (
        "_query_leader still uses wait_for(open_connection(...)); switch to "
        "asyncio.timeout for cancel-scope semantics that don't discard "
        "the inner-task result on outer-cancel"
    )
    assert "asyncio.timeout(self._dial_timeout)" in src, (
        "_query_leader must use asyncio.timeout(self._dial_timeout) for the dial"
    )


def test_query_leader_writer_initialised_before_try() -> None:
    """``writer = None`` is initialised before the try so the finally
    always sees a defined name (no NameError on cancel-before-dial)."""
    src = _source_of("_query_leader")
    assert "writer = None" in src, (
        "_query_leader must initialise writer = None before the try-block"
    )
    pos_init = src.index("writer = None")
    pos_timeout = src.index("asyncio.timeout(self._dial_timeout)")
    assert pos_init < pos_timeout, (
        "writer = None must appear BEFORE the asyncio.timeout block "
        "(initialised pre-try so the finally always sees a defined name)"
    )


def test_open_admin_connection_no_longer_uses_wait_for_around_open_connection() -> None:
    """Symmetric pin for ``open_admin_connection`` (the public admin-
    protocol context manager)."""
    src = _source_of("open_admin_connection")
    src_stripped = src.replace(" ", "").replace("\n", "")
    assert "wait_for(open_connection(" not in src_stripped, (
        "open_admin_connection still uses wait_for(open_connection(...)); switch to asyncio.timeout"
    )
    assert "asyncio.timeout(self._dial_timeout)" in src, (
        "open_admin_connection must use asyncio.timeout(self._dial_timeout)"
    )


def test_connect_impl_no_longer_uses_wait_for_around_open_connection() -> None:
    """Same pin for ``DqliteConnection._connect_impl``, the path every
    user-level connect goes through."""
    src = _connection_source_of("_connect_impl")
    src_stripped = src.replace(" ", "").replace("\n", "")
    assert "wait_for(open_connection(" not in src_stripped, (
        "_connect_impl still uses wait_for(open_connection(...)); switch to "
        "asyncio.timeout for cancel-scope semantics that don't discard "
        "the inner-task result on outer-cancel"
    )
    assert "asyncio.timeout(self._dial_timeout)" in src, (
        "_connect_impl must use asyncio.timeout(self._dial_timeout) for the dial"
    )


def test_connect_impl_writer_nulled_after_protocol_handoff() -> None:
    """After ``DqliteProtocol(...)`` takes ownership, ``writer`` is
    nulled so the outer finally drain cannot double-close it."""
    src = _connection_source_of("_connect_impl")
    proto_pos = src.find("DqliteProtocol(")
    null_pos = src.find("writer = None", proto_pos)
    assert proto_pos != -1
    assert null_pos != -1, (
        "_connect_impl must null writer AFTER the DqliteProtocol(...) "
        "constructor returns — the protocol now owns the transport, and "
        "a stray writer reference in the local frame turns the outer "
        "finally drain into a double-close on any future protocol-side "
        "close path."
    )


def test_open_admin_connection_writer_owned_by_outer_finally() -> None:
    """``open_admin_connection`` must NOT null ``writer`` after the
    hand-off: it yields the protocol, so the outer finally is the only
    drain path; nulling would orphan the writer on success."""
    src = _source_of("open_admin_connection")
    proto_pos = src.find("DqliteProtocol(")
    negotiate_pos = src.find("await protocol.negotiate_protocol_only")
    assert proto_pos != -1
    assert negotiate_pos != -1
    between = src[proto_pos:negotiate_pos]
    assert "writer = None" not in between, (
        "open_admin_connection must NOT null writer between the "
        "DqliteProtocol(...) hand-off and the version negotiation — "
        "the protocol is yielded (not stored), so the outer finally "
        "is the only drain path. Nulling here would orphan the writer "
        "on success."
    )


def test_connect_impl_writer_initialised_before_try() -> None:
    """``writer = None`` is initialised before the try so the finally
    always sees a defined name (mirrors the ``_query_leader`` pin)."""
    src = _connection_source_of("_connect_impl")
    assert "writer = None" in src, (
        "_connect_impl must initialise writer = None before the outer try-block"
    )
    pos_init = src.index("writer = None")
    pos_timeout = src.index("asyncio.timeout(self._dial_timeout)")
    assert pos_init < pos_timeout, (
        "writer = None must appear BEFORE the asyncio.timeout block in "
        "_connect_impl (initialised pre-try so the finally always sees a "
        "defined name)"
    )
