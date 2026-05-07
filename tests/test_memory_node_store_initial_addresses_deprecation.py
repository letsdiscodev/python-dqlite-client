"""Pin: ``MemoryNodeStore(initial_addresses=...)`` emits a
``DeprecationWarning`` so callers see the migration roadmap to
``addresses=...`` rather than only the docstring tag.

The constructor accepts both ``addresses`` (preferred) and
``initial_addresses`` (deprecated). The docstring marks the latter
deprecated; this pin ensures the runtime signal exists too, with
``stacklevel=2`` so the warning points at the caller's
``MemoryNodeStore(...)`` line, not the line in ``__init__`` itself.

The stacklevel pin matters: an implementer who sets
``stacklevel=1`` (or omits it) would still pass a basic count-and-
category assertion, but the warning would surface from
``node_store.py`` rather than the user's call site — defeating the
operator's "find every deprecated usage" grep over the test
suite.
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
    """The warning must point at the caller's invocation line, not at
    the line inside ``MemoryNodeStore.__init__``. Verified by the
    captured ``filename`` matching this test file rather than
    ``node_store.py``.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore(initial_addresses=["leader:9001"])

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    # ``stacklevel=2`` makes the warning appear to come from this
    # test file. ``stacklevel=1`` (default) would surface
    # ``node_store.py``; that's what this assertion fences against.
    assert deprecations[0].filename.endswith(
        "test_memory_node_store_initial_addresses_deprecation.py"
    ), (
        f"DeprecationWarning surfaced from {deprecations[0].filename}; "
        f"expected the test file (stacklevel=2 not honoured)"
    )


def test_addresses_kwarg_emits_no_warning() -> None:
    """Positive control: the preferred kwarg path stays silent."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore(addresses=["leader:9001"])

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []


def test_no_args_emits_no_warning() -> None:
    """Empty-construction path stays silent (no kwarg used at all)."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        MemoryNodeStore()

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []


def test_initial_addresses_path_still_seeds_the_store() -> None:
    """Functional regression check: the deprecated kwarg still
    populates the store. Run under ``filterwarnings`` to ignore the
    expected DeprecationWarning so a global ``-W error`` test
    configuration doesn't flip this into a failure."""
    import asyncio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store = MemoryNodeStore(initial_addresses=["leader:9001"])

    nodes = asyncio.run(store.get_nodes())
    assert len(nodes) == 1
    assert nodes[0].address == "leader:9001"


@pytest.mark.parametrize("kwarg", ["initial_addresses", "addresses"])
def test_both_kwargs_rejected_before_warning(kwarg: str) -> None:
    """The TypeError for "both kwargs" must fire BEFORE any warning,
    regardless of which kwarg name comes second. Pinning ordering so
    a future refactor that warns first then raises does not flip the
    user-visible behaviour."""
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
