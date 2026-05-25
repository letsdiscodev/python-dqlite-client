"""``ClusterClient.__init__`` must validate ``max_total_rows``,
``max_continuation_frames`` and ``max_message_size`` at construction
— matching the discipline at ``DqliteConnection.__init__`` — so a
misconfigured value surfaces at the operator's config-load site
rather than deep inside a per-node probe coroutine on the first
``find_leader()``.

The pre-fix behaviour stored the value verbatim on ``self._max_*`` and
only ran the validator at ``DqliteProtocol.__init__``, which a probe
coroutine constructs many RPCs later. A ``bool`` (e.g.
``max_total_rows=True``) would silently propagate through every
admin and probe RPC for the lifetime of the instance until the first
construction tripped the validator inside ``_query_leader``.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.node_store import MemoryNodeStore


def _store() -> MemoryNodeStore:
    return MemoryNodeStore(["127.0.0.1:9001"])


@pytest.mark.parametrize("kwarg", ["max_total_rows", "max_continuation_frames"])
def test_cluster_client_rejects_zero_int_caps_at_construction(kwarg: str) -> None:
    kwargs: dict[str, object] = {kwarg: 0}
    with pytest.raises(ValueError, match=f"{kwarg} must be > 0"):
        ClusterClient(_store(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwarg", ["max_total_rows", "max_continuation_frames"])
def test_cluster_client_rejects_negative_int_caps_at_construction(kwarg: str) -> None:
    kwargs: dict[str, object] = {kwarg: -5}
    with pytest.raises(ValueError, match=f"{kwarg} must be > 0"):
        ClusterClient(_store(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwarg", ["max_total_rows", "max_continuation_frames"])
def test_cluster_client_rejects_bool_caps_at_construction(kwarg: str) -> None:
    kwargs: dict[str, object] = {kwarg: True}
    with pytest.raises(TypeError, match=f"{kwarg} must be int or None"):
        ClusterClient(_store(), **kwargs)  # type: ignore[arg-type]


def test_cluster_client_max_message_size_zero_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="max_message_size must be >= 1"):
        ClusterClient(_store(), max_message_size=0)


def test_cluster_client_max_message_size_negative_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="max_message_size must be >= 1"):
        ClusterClient(_store(), max_message_size=-1)


def test_cluster_client_max_message_size_bool_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="max_message_size must be int or None"):
        ClusterClient(_store(), max_message_size=True)


@pytest.mark.parametrize(
    "kwarg",
    ["max_total_rows", "max_continuation_frames", "max_message_size"],
)
def test_cluster_client_none_allowed_for_caps(kwarg: str) -> None:
    # None disables the cap (or defers to the wire-layer default) and
    # must be accepted at construction.
    kwargs: dict[str, object] = {kwarg: None}
    ClusterClient(_store(), **kwargs)  # type: ignore[arg-type]
