"""Pin: ``ClusterClient._query_leader``, ``_acquire_admin_protocol``,
and ``DqliteConnection._connect_impl`` do not orphan
``(reader, writer)`` when an outer cancel lands during the dial.

The pre-fix shape used ``asyncio.wait_for(open_connection(...))`` —
documented CPython hazard: when the inner ``open_connection`` task
resolves with ``(reader, writer)`` but the outer ``wait_for`` is
cancelled before unpack, the result is discarded and the writer
never reaches a name. Under leader-probe stampede with an outer
``asyncio.timeout``, every cancelled probe leaks one socket + reader
Task (FIN-WAIT-2 + ResourceWarning at GC).

The fix replaces ``wait_for`` with ``async with asyncio.timeout``
(cancel-scope semantics; does NOT discard the inner result on
outer-cancel) and moves the dial INSIDE the outer try-block so the
finally drain runs on every exit arm. ``writer = None`` is
initialized before the try; the finally guards on
``writer is not None``.

Pin shape (structural / source-level): assert the source no longer
contains ``await asyncio.wait_for(open_connection(...))`` patterns.
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
    """``_query_leader`` must use ``asyncio.timeout(...)`` (cancel-
    scope semantics) instead of ``asyncio.wait_for(open_connection(
    ...))`` for the dial. The latter has documented CPython behavior
    of discarding the inner result on outer-cancel — the canonical
    leak shape this fix is closing.
    """
    src = _source_of("_query_leader")
    # The test guard: the wait_for-around-open_connection composite
    # should NOT appear in the source. ``asyncio.timeout`` is fine
    # (it has cancel-scope semantics). Strip whitespace from BOTH the
    # source AND the search literal so the pattern compares apples-
    # to-apples regardless of the indentation depth in the production
    # code. Earlier shape stripped only the source; the literal still
    # carried 12 spaces, so the ``not in`` was vacuously satisfied —
    # the guard would have passed against the regression it claimed
    # to fence.
    src_stripped = src.replace(" ", "").replace("\n", "")
    assert "wait_for(open_connection(" not in src_stripped, (
        "_query_leader still uses wait_for(open_connection(...)); switch to "
        "asyncio.timeout for cancel-scope semantics that don't discard "
        "the inner-task result on outer-cancel"
    )
    # Also assert the asyncio.timeout pattern IS present (the fix shape).
    assert "asyncio.timeout(self._dial_timeout)" in src, (
        "_query_leader must use asyncio.timeout(self._dial_timeout) for the dial"
    )


def test_query_leader_writer_initialised_before_try() -> None:
    """``writer = None`` must be initialised before the outer try-
    block so the finally always sees a defined name (avoids
    NameError on the cancel-before-dial-completes path).
    """
    src = _source_of("_query_leader")
    # Find the ``writer = None`` initialization before the ``try:``
    # block that wraps the dial.
    assert "writer = None" in src, (
        "_query_leader must initialise writer = None before the try-block"
    )
    # Order check: ``writer = None`` appears before ``async with asyncio.timeout``.
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
    # Same whitespace-normalisation fix as in the sibling test —
    # earlier shape stripped only the source while the search literal
    # carried indentation, making the assertion vacuous.
    src_stripped = src.replace(" ", "").replace("\n", "")
    assert "wait_for(open_connection(" not in src_stripped, (
        "open_admin_connection still uses wait_for(open_connection(...)); switch to asyncio.timeout"
    )
    assert "asyncio.timeout(self._dial_timeout)" in src, (
        "open_admin_connection must use asyncio.timeout(self._dial_timeout)"
    )


def test_connect_impl_no_longer_uses_wait_for_around_open_connection() -> None:
    """Third call site: ``DqliteConnection._connect_impl`` (the path
    every user-level connect goes through). The cluster-side
    ``_query_leader`` / ``open_admin_connection`` got the
    ``asyncio.timeout``-based fix, but the connection-side
    ``_connect_impl`` was left on the leaky composite; an outer
    cancel landing while the dial is in flight under
    ``DqliteConnection.connect()`` would discard the result.
    """
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
    """Hand-off discipline: after ``DqliteProtocol(reader, writer, ...)``
    constructs (the protocol now owns the transport), null
    ``writer`` so the outer finally drain becomes a no-op for the
    success path. Without this null, a future refactor that adds a
    protocol-side close in the same scope would double-close the
    underlying writer."""
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
    """``open_admin_connection`` deliberately does NOT null ``writer``
    after the ``DqliteProtocol(...)`` hand-off. Unlike
    ``_connect_impl`` (where the protocol is stored on ``self`` and a
    later ``close()`` drains the writer via the protocol-owned
    reference), this method ``yield``-s the protocol; the outer
    ``finally`` is the only place the writer can be closed. Nulling
    after the hand-off would orphan the writer on the success path.

    This pin documents the deliberate divergence from the
    ``_connect_impl`` discipline and fences against a future
    "consistency" refactor introducing the leak.
    """
    src = _source_of("open_admin_connection")
    proto_pos = src.find("DqliteProtocol(")
    # open_admin_connection migrated from the full handshake() to the
    # lighter negotiate_protocol_only() (go-parity with
    # NewDirectConnector.Connect); the writer-ownership invariant the
    # original pin targets is unchanged.
    negotiate_pos = src.find("await protocol.negotiate_protocol_only")
    assert proto_pos != -1
    assert negotiate_pos != -1
    # No `writer = None` between DqliteProtocol(...) and
    # `await protocol.negotiate_protocol_only` — that's the load-bearing
    # divergence.
    between = src[proto_pos:negotiate_pos]
    assert "writer = None" not in between, (
        "open_admin_connection must NOT null writer between the "
        "DqliteProtocol(...) hand-off and the version negotiation — "
        "the protocol is yielded (not stored), so the outer finally "
        "is the only drain path. Nulling here would orphan the writer "
        "on success."
    )


def test_connect_impl_writer_initialised_before_try() -> None:
    """``writer = None`` must be initialised before the outer try-
    block so the finally always sees a defined name (avoids
    NameError on the cancel-before-dial-completes path).

    Mirrors the pin already enforced for ``_query_leader``.
    """
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
