"""Pin: ``DqliteConnection.__repr__`` routes ``self._address`` through
``sanitize_for_log``. Python's ``repr()`` escapes ASCII C0 but not
U+2028/U+2029/bidi/zero-width/BOM, which journald would split on.
"""

from __future__ import annotations

from dqliteclient.connection import DqliteConnection


def _synth_conn(address: str) -> DqliteConnection:
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
    # U+202E (RLO) is not escaped by repr() and can rewrite logs.
    conn = _synth_conn("host‮9001")
    rendered = repr(conn)
    assert "‮" not in rendered


def test_repr_replaces_zwsp() -> None:
    # U+200B (ZWSP) is invisible and not escaped by repr().
    conn = _synth_conn("host​:9001")
    rendered = repr(conn)
    assert "​" not in rendered


def test_repr_escapes_ascii_lf() -> None:
    # LF was already escaped by repr(); pin that the sanitiser doesn't undo it.
    conn = _synth_conn("evil\nhost:9001")
    rendered = repr(conn)
    assert "\n" not in rendered
    assert "\\n" in rendered or "?" in rendered


def test_repr_ascii_host_port_unchanged() -> None:
    """ASCII host:port survives byte-for-byte (sanitiser is a no-op here)."""
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
