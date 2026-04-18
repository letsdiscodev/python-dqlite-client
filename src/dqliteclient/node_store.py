"""Node store interfaces for cluster discovery."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a cluster node.

    Frozen so callers holding a reference from ``get_nodes()`` can't
    accidentally mutate store state (and so instances are hashable
    and usable as set/dict keys for deduplication).
    """

    node_id: int
    address: str
    role: int  # 0=voter, 1=standby, 2=spare


class NodeStore(ABC):
    """Abstract interface for storing cluster node information."""

    @abstractmethod
    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Get list of known nodes."""
        ...

    @abstractmethod
    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes."""
        ...


class MemoryNodeStore(NodeStore):
    """In-memory node store.

    Backs storage with an immutable tuple so readers can safely be
    handed a direct reference without defensive copies and so a
    concurrent ``set_nodes()`` never produces a torn view.
    """

    def __init__(self, initial_addresses: list[str] | None = None) -> None:
        if initial_addresses:
            self._nodes: tuple[NodeInfo, ...] = tuple(
                NodeInfo(node_id=i + 1, address=addr, role=0)
                for i, addr in enumerate(initial_addresses)
            )
        else:
            self._nodes = ()

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Get list of known nodes."""
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes."""
        self._nodes = tuple(nodes)
