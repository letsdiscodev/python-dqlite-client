"""Node store interfaces for cluster discovery."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NodeInfo:
    """Information about a cluster node."""

    node_id: int
    address: str
    role: int  # 0=spare, 1=voter, 2=standby


class NodeStore(ABC):
    """Abstract interface for storing cluster node information."""

    @abstractmethod
    async def get_nodes(self) -> list[NodeInfo]:
        """Get list of known nodes."""
        ...

    @abstractmethod
    async def set_nodes(self, nodes: list[NodeInfo]) -> None:
        """Update list of known nodes."""
        ...


class MemoryNodeStore(NodeStore):
    """In-memory node store."""

    def __init__(self, initial_addresses: list[str] | None = None) -> None:
        self._nodes: list[NodeInfo] = []
        if initial_addresses:
            for i, addr in enumerate(initial_addresses):
                self._nodes.append(NodeInfo(node_id=i, address=addr, role=1))

    async def get_nodes(self) -> list[NodeInfo]:
        """Get list of known nodes."""
        return list(self._nodes)

    async def set_nodes(self, nodes: list[NodeInfo]) -> None:
        """Update list of known nodes."""
        self._nodes = list(nodes)
