"""Pin: a dangling symlink at the store path is NOT silently treated
as "file missing". ``Path.exists()`` follows symlinks and returns
False for a dangling symlink target; the pre-fix existence-check arm
therefore short-circuited to an empty-store fallback, pre-empting the
``O_NOFOLLOW`` guard the open-once path documents.

Replacing ``Path.exists()`` with ``Path.lstat()`` (does NOT follow
symlinks) restores the discipline: only a true "no entry at this
path" condition collapses to the empty-store fallback; everything
else (including a dangling symlink) reaches the ``os.open(O_NOFOLLOW)``
call which raises ``ELOOP`` or ``ENOENT`` as appropriate.
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
    """The symlink target does not exist. ``Path.exists()`` returns
    False (silently treats as missing) — but the operator would
    expect to see the configuration error rather than a silent empty
    store. After the fix, ``lstat`` succeeds and ``os.open(O_NOFOLLOW)``
    surfaces the dangling-target condition as a ClusterError."""
    nonexistent = tmp_path / "nonexistent.yaml"
    link = tmp_path / "link.yaml"
    link.symlink_to(nonexistent)

    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = link
    store._creator_pid = os.getpid()

    with pytest.raises(ClusterError, match="(?i)cannot open"):
        store._load_from_disk()


def test_genuinely_missing_file_still_returns_empty(tmp_path: Path) -> None:
    """Sibling positive: a path with no filesystem entry continues
    to collapse to the empty-store fallback (the documented
    go-dqlite parity behaviour for ``NewYamlNodeStore`` at startup)."""
    missing = tmp_path / "absent.yaml"

    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = missing
    store._creator_pid = os.getpid()

    assert store._load_from_disk() == ()
