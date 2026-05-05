"""``DqliteConnection.__repr__`` must include ``id(self)`` so two
connections to the same ``host:port`` are visually distinguishable
in logs.

Every other repr in the family (``Cursor``, ``AsyncCursor``,
``Connection``, ``AsyncConnection``, ``ConnectionPool``) carries
``at 0x{id:x}>``; the ``DqliteConnection`` repr was the lone
exception.
"""

from dqliteclient.connection import DqliteConnection


def test_dqlite_connection_repr_includes_id() -> None:
    a = DqliteConnection("127.0.0.1:9001", database="db")
    b = DqliteConnection("127.0.0.1:9001", database="db")

    repr_a = repr(a)
    repr_b = repr(b)

    assert " at 0x" in repr_a, repr_a
    assert " at 0x" in repr_b, repr_b
    assert hex(id(a))[2:] in repr_a
    assert hex(id(b))[2:] in repr_b
    assert repr_a != repr_b


def test_dqlite_connection_repr_state_changes_visible() -> None:
    conn = DqliteConnection("127.0.0.1:9001", database="db")

    r = repr(conn)
    assert "disconnected" in r
    assert "127.0.0.1:9001" in r
    assert " at 0x" in r
