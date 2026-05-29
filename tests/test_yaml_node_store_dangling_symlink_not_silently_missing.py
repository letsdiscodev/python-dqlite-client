"""Pin: a dangling symlink is NOT silently treated as "file missing".

``Path.exists()`` follows symlinks (False for a dangling target), short-circuiting to
the empty-store fallback; ``Path.lstat()`` doesn't follow, so only a true missing entry
collapses to empty and everything else reaches the ``O_NOFOLLOW`` guard.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import YamlNodeStore


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"),
    reason="O_NOFOLLOW unavailable (Windows fallback path)",
)
def test_dangling_symlink_surfaces_as_cluster_error(tmp_path: Path) -> None:
    """A dangling symlink must surface as a config error, not a silent empty store."""
    nonexistent = tmp_path / "nonexistent.yaml"
    link = tmp_path / "link.yaml"
    link.symlink_to(nonexistent)

    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = link
    store._creator_pid = os.getpid()

    with pytest.raises(ClusterError, match="(?i)cannot open"):
        store._load_from_disk()


def test_genuinely_missing_file_still_returns_empty(tmp_path: Path) -> None:
    """A path with no filesystem entry still collapses to the empty-store fallback
    (go-dqlite parity)."""
    missing = tmp_path / "absent.yaml"

    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = missing
    store._creator_pid = os.getpid()

    assert store._load_from_disk() == ()
