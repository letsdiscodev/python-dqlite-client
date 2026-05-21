"""Pin: ``DqliteConnection.__repr__`` routes ``self._address``
through ``sanitize_for_log`` before the ``%r``-style interpolation.

Python's ``repr()`` of a ``str`` escapes the canonical control set
(LF / CR / Tab / NUL / ASCII C0) but NOT U+2028 / U+2029 / bidi /
zero-width / BOM. journald treats U+2028 as a record separator, so
a peer-supplied address smuggled past ``parse_address`` (a future
custom ``dial_func`` or leader-tracker refactor are the two
documented scenarios) would split a downstream
``logger.X("%r", conn)`` record without this wrap.

Defence-in-depth: ``parse_address``'s strict gate is the in-tree
invariant blocking exploit today; centralising sanitisation inside
``__repr__`` covers third-party ``%r`` consumers (asyncio Task
repr, SA pool repr with ``echo_pool=True``) without per-call-site
wrapping.
"""

from __future__ import annotations

from dqliteclient.connection import DqliteConnection


def _synth_conn(address: str) -> DqliteConnection:
    """Build a stub-shaped DqliteConnection just for repr."""
    conn = DqliteConnection.__new__(DqliteConnection)
    conn._address = address
    conn._database = "main"
    conn._protocol = None
    return conn


def test_repr_replaces_u2028_line_separator() -> None:
    conn = _synth_conn("leader forged:9001")
    rendered = repr(conn)
    assert " " not in rendered, (
        "U+2028 must be replaced inside the repr; Python's repr() "
        "alone leaves it raw and journald splits the log line."
    )


def test_repr_replaces_u2029_paragraph_separator() -> None:
    conn = _synth_conn("leader forged:9001")
    rendered = repr(conn)
    assert " " not in rendered


def test_repr_replaces_bidi_rlo() -> None:
    # U+202E (RIGHT-TO-LEFT OVERRIDE) is not escaped by repr() and
    # can rewrite operator-facing logs.
    conn = _synth_conn("host‮9001")
    rendered = repr(conn)
    assert "‮" not in rendered


def test_repr_replaces_zwsp() -> None:
    # U+200B (ZERO WIDTH SPACE) is invisible and not escaped by repr().
    conn = _synth_conn("host​:9001")
    rendered = repr(conn)
    assert "​" not in rendered


def test_repr_escapes_ascii_lf() -> None:
    # LF was already escaped by repr() pre-fix; pin as a regression
    # guard (the sanitiser must not undo this).
    conn = _synth_conn("evil\nhost:9001")
    rendered = repr(conn)
    assert "\n" not in rendered
    assert "\\n" in rendered or "?" in rendered


def test_repr_ascii_host_port_unchanged() -> None:
    """ASCII host:port must survive byte-for-byte — the sanitiser
    is a no-op on the safe subset."""
    conn = _synth_conn("leader.example.com:9001")
    rendered = repr(conn)
    assert "leader.example.com:9001" in rendered
    assert rendered.startswith("<DqliteConnection address=")
    assert rendered.endswith(">")


def test_repr_includes_connection_state() -> None:
    conn_disc = _synth_conn("host:9001")
    assert "disconnected" in repr(conn_disc)


def test_repr_includes_id() -> None:
    conn = _synth_conn("host:9001")
    rendered = repr(conn)
    assert f"0x{id(conn):x}" in rendered
