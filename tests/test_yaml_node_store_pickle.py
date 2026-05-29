"""Pin: ``YamlNodeStore`` is picklable (it must not cache the yaml module)."""

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
        data = pickle.dumps(store)
        restored = pickle.loads(data)
        assert restored.path == store.path
    finally:
        os.unlink(path)


def test_yaml_node_store_does_not_cache_module_reference() -> None:
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
