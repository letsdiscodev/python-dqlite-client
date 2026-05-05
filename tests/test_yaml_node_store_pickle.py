"""Pin: ``YamlNodeStore`` is picklable.

Previously the class cached the imported ``yaml`` module on
``self._yaml`` (deferred import for the optional dependency). Module
objects are not picklable; ``pickle.dumps(store)`` failed with
``TypeError: cannot pickle 'module' object`` — no class name, no
remediation hint, asymmetric with sibling ``MemoryNodeStore`` which
pickles cleanly.

Re-importing on demand inside ``_load_from_disk`` / ``_save_to_disk``
removes the unpicklable module reference; cached ``import yaml``
costs ~µs per call site.
"""

import os
import pickle
import tempfile

import pytest

pytest.importorskip("yaml")

from dqliteclient.node_store import YamlNodeStore  # noqa: E402


def test_yaml_node_store_round_trip_pickle() -> None:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        os.close(fd)
        store = YamlNodeStore(path)
        # Confirm pickle round-trip works.
        data = pickle.dumps(store)
        restored = pickle.loads(data)
        assert restored.path == store.path
    finally:
        os.unlink(path)


def test_yaml_node_store_does_not_cache_module_reference() -> None:
    """Pin: the class no longer caches the yaml module on the
    instance. Caching it broke pickle round-trip; the cleaner
    fix re-imports lazily at the two call sites."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        os.close(fd)
        store = YamlNodeStore(path)
        assert not hasattr(store, "_yaml"), (
            "YamlNodeStore should not cache the yaml module on self; "
            "re-import on demand instead so pickle round-trip works."
        )
    finally:
        os.unlink(path)
