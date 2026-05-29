"""``MemoryNodeStore(initial_addresses=...)`` emits a DeprecationWarning.

``stacklevel=2`` must point the warning at the caller, not ``node_store.py``.
"""

from __future__ import annotations

import warnings

import pytest

from dqliteclient.node_store import MemoryNodeStore


def test_initial_addresses_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore(initial_addresses=["leader:9001"])

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, (
        f"expected exactly one DeprecationWarning, got {len(deprecations)}: "
        f"{[str(w.message) for w in deprecations]}"
    )
    assert "initial_addresses" in str(deprecations[0].message)
    assert "addresses" in str(deprecations[0].message)


def test_initial_addresses_warning_stacklevel_points_at_caller() -> None:
    """The warning points at the caller's line, not inside __init__."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore(initial_addresses=["leader:9001"])

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert deprecations[0].filename.endswith(
        "test_memory_node_store_initial_addresses_deprecation.py"
    ), (
        f"DeprecationWarning surfaced from {deprecations[0].filename}; "
        f"expected the test file (stacklevel=2 not honoured)"
    )


def test_addresses_kwarg_emits_no_warning() -> None:
    """The preferred kwarg path stays silent."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore(addresses=["leader:9001"])

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []


def test_no_args_emits_no_warning() -> None:
    """Empty-construction path stays silent."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore()

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []


def test_initial_addresses_path_still_seeds_the_store() -> None:
    """The deprecated kwarg still populates the store."""
    import asyncio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store = MemoryNodeStore(initial_addresses=["leader:9001"])

    nodes = asyncio.run(store.get_nodes())
    assert len(nodes) == 1
    assert nodes[0].address == "leader:9001"


@pytest.mark.parametrize("kwarg", ["initial_addresses", "addresses"])
def test_both_kwargs_rejected_before_warning(kwarg: str) -> None:
    """The "both kwargs" TypeError fires before any warning, regardless of kwarg order."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="Pass only one"):
            MemoryNodeStore(addresses=["a:1"], initial_addresses=["b:2"])

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], (
        f"deprecation warning fired during the TypeError path; should be "
        f"silent because the TypeError is the user-visible signal: "
        f"{[str(w.message) for w in deprecations]}"
    )
