"""Client-layer closed-value dataclasses carry ``@final``.

Subclassing a ``frozen=True, slots=True`` dataclass silently breaks ``__slots__``
inheritance; ``@final`` surfaces the attempt at static-analysis time.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import LeaderInfo, NodeMetadata
from dqliteclient.node_store import NodeInfo as ClientNodeInfo


@pytest.mark.parametrize(
    "cls",
    [LeaderInfo, NodeMetadata, ClientNodeInfo],
    ids=["LeaderInfo", "NodeMetadata", "node_store.NodeInfo"],
)
def test_client_value_dataclass_is_final(cls: type) -> None:
    assert getattr(cls, "__final__", False) is True
