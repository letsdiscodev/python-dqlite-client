"""Pin: the ``trust_server_heartbeat`` paragraph in
:meth:`ClusterClient.__init__` no longer uses internal-tracker
vocabulary (``TODO``) inside its user-facing docstring.

The earlier phrasing parenthesised ``whose heartbeat is
TODO/disabled``, which is ambiguous between three readings (project
TODO marker, go-dqlite pending feature, or go-dqlite design choice).
Replaced with an explicit description of go-dqlite's actual
behaviour (it advertises the heartbeat but does not adjust read
deadlines on it) so operators making the ``trust_server_heartbeat``
defaults-vs-opt-in decision get a load-bearing rationale rather
than a cipher.

A grep guard at the docstring level catches a regression that would
re-introduce ``TODO`` (or sibling tracker tokens) inside the
``trust_server_heartbeat`` paragraph specifically.
"""

from __future__ import annotations

from dqliteclient.cluster import ClusterClient


def test_cluster_init_docstring_has_no_todo_token() -> None:
    doc = ClusterClient.__init__.__doc__
    assert doc is not None
    assert "TODO" not in doc
    assert "FIXME" not in doc
    assert "XXX" not in doc
    assert "HACK" not in doc


def test_cluster_init_docstring_keeps_go_dqlite_cross_reference() -> None:
    """The replacement phrasing must keep the ``go-dqlite`` anchor
    so readers can still cross-check the parity rationale for the
    ``False`` default.
    """
    doc = ClusterClient.__init__.__doc__
    assert doc is not None
    assert "go-dqlite" in doc
    assert "trust_server_heartbeat" in doc
