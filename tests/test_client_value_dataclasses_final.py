"""Pin: client-layer closed-value dataclasses carry ``@final``.

The wire-layer ``*Response`` classes plus
``dqlitewire.messages.responses.NodeInfo`` were decorated with
``@final`` to surface subclass attempts at static-analysis time
(subclassing a ``frozen=True, slots=True`` dataclass silently breaks
``__slots__`` inheritance unless the subclass redeclares slots, and
``__post_init__`` overrides bypass the role-validation guard on
``NodeInfo``).

The client-layer dataclasses with the identical closed-value
contract — ``cluster.LeaderInfo``, ``cluster.NodeMetadata``, and
``node_store.NodeInfo`` — were missed in that landing. This pin
keeps the discipline mirrored across the wire/client boundary.

Probe via ``getattr(cls, "__final__", False)`` for forward
compatibility with the ``typing.final`` runtime marker (set on the
wrapped class object on Python 3.11+).
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
