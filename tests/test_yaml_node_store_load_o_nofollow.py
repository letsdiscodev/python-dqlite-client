"""Pin: ``YamlNodeStore._load_from_disk`` opens the file once with
``O_NOFOLLOW`` and reads bounded bytes from the open fd. Defends
against:
- CWE-367 TOCTOU: a co-tenant rename / inode-swap between a
  separate ``stat()`` + ``read_text()`` could have allowed reading
  bytes from a different inode than the one whose size was
  checked.
- CWE-59 symlink-follow: a symlink at the target path would have
  been followed silently by ``read_text``; ``O_NOFOLLOW`` rejects
  it with ``ELOOP``.
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
def test_load_from_disk_rejects_symlink(tmp_path: Path) -> None:
    """A symlink at the target path must be rejected by O_NOFOLLOW
    rather than followed silently. Pre-fix, ``Path.read_text``
    followed the symlink — a co-tenant could swap a small stat'd
    file for a symlink to /dev/zero between stat and read."""
    real = tmp_path / "real.yaml"
    real.write_text("- {ID: 1, Address: '127.0.0.1:9001', Role: 0}\n")
    link = tmp_path / "link.yaml"
    link.symlink_to(real)

    # The store points at the symlink. O_NOFOLLOW raises ELOOP.
    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = link
    import os as _os

    store._creator_pid = _os.getpid()
    # Re-implement the constructor's bare-minimum init via __new__
    # without dialing — this test only exercises the load path.
    with pytest.raises(ClusterError, match="(?i)cannot open"):
        store._load_from_disk()


def test_load_from_disk_rejects_oversize_growth_post_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the inode-swap race: the file fstat reports a
    sub-cap size, but the read returns more bytes than the cap (a
    racing writer grew the file between fstat and read). The cap+1
    bounded read MUST detect the overflow and refuse."""
    from dqliteclient import node_store as ns_mod

    real = tmp_path / "real.yaml"
    real.write_text("- {ID: 1, Address: '127.0.0.1:9001', Role: 0}\n")

    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = real
    store._creator_pid = os.getpid()

    # Monkeypatch os.read to return cap+2 bytes regardless of input.
    real_read = os.read
    cap_plus_two = b"a" * (ns_mod._MAX_YAML_NODE_STORE_BYTES + 2)

    def _bogus_read(fd: int, n: int) -> bytes:
        # Restore so the close() doesn't double-bogus.
        return cap_plus_two

    monkeypatch.setattr("dqliteclient.node_store.os.read", _bogus_read)
    try:
        with pytest.raises(ClusterError, match="(?i)grew past"):
            store._load_from_disk()
    finally:
        monkeypatch.setattr("dqliteclient.node_store.os.read", real_read)


def test_load_from_disk_happy_path(tmp_path: Path) -> None:
    """Negative pin: a small legitimate file still loads correctly
    after the open-once + fstat + bounded-read migration."""
    target = tmp_path / "nodes.yaml"
    target.write_text(
        "- {ID: 1, Address: '127.0.0.1:9001', Role: 0}\n"
        "- {ID: 2, Address: '127.0.0.1:9002', Role: 0}\n"
    )

    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = target
    store._creator_pid = os.getpid()
    nodes = store._load_from_disk()
    assert len(nodes) == 2
    assert nodes[0].address == "127.0.0.1:9001"
    assert nodes[1].address == "127.0.0.1:9002"


def test_load_from_disk_rejects_non_utf8(tmp_path: Path) -> None:
    """Open-once path explicitly decodes bytes; a non-UTF-8 file
    surfaces as ClusterError instead of bare UnicodeDecodeError."""
    target = tmp_path / "nodes.yaml"
    target.write_bytes(b"\xff\xfe not utf-8")
    store = YamlNodeStore.__new__(YamlNodeStore)
    store._path = target
    store._creator_pid = os.getpid()
    with pytest.raises(ClusterError, match="(?i)non-utf-8"):
        store._load_from_disk()
