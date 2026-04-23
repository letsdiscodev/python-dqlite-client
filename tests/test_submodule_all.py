"""Each dqliteclient submodule must declare ``__all__`` so
``from dqliteclient.<sub> import *`` does not leak private helpers.

Mirrors the pattern in python-dqlite-dbapi (see
`test_cursor_module_missing_all` and friends).
"""

from __future__ import annotations

import importlib

import pytest

_SUBMODULES = [
    "dqliteclient.cluster",
    "dqliteclient.connection",
    "dqliteclient.exceptions",
    "dqliteclient.node_store",
    "dqliteclient.pool",
    "dqliteclient.protocol",
    "dqliteclient.retry",
]


@pytest.mark.parametrize("modname", _SUBMODULES)
def test_submodule_declares_all(modname: str) -> None:
    mod = importlib.import_module(modname)
    assert hasattr(mod, "__all__"), f"{modname} is missing __all__"
    exported = mod.__all__  # type: ignore[attr-defined]
    assert isinstance(exported, list | tuple), (
        f"{modname}.__all__ must be list/tuple, got {type(exported).__name__}"
    )
    for name in exported:
        assert isinstance(name, str), f"{modname}.__all__ entries must be strings; got {name!r}"
        assert hasattr(mod, name), f"{modname}.__all__ lists {name!r} but it is not defined"
