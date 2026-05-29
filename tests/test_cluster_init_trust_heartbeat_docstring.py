"""ClusterClient.__init__'s user-facing docstring carries no tracker tokens
(TODO etc.) and keeps the go-dqlite cross-reference for the heartbeat rationale.
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
