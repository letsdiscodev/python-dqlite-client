"""Node store interfaces for cluster discovery."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from dqlitewire import NodeRole

__all__ = ["MemoryNodeStore", "NodeInfo", "NodeStore"]


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a cluster node.

    Frozen so callers holding a reference from ``get_nodes()`` can't
    accidentally mutate store state (and so instances are hashable
    and usable as set/dict keys for deduplication).
    """

    node_id: int
    """Dqlite raft node id. Typically 1-based, assigned by the
    cluster when the node joined. In the
    ``MemoryNodeStore.__init__(addresses=[...])`` seeding path the id
    is synthesised (``i + 1``) and does NOT correspond to the
    cluster-side id — it's a placeholder used for ordering and
    dedup. Callers that need the authoritative id should populate
    the store from a control-plane source that knows it."""

    address: str
    """``host:port`` string. IPv6 addresses must be bracketed
    (``[::1]:9000``). Parsed by
    ``dqliteclient.connection._parse_address`` — hostnames are
    ASCII-only and lowercased, IP literals are canonicalised."""

    role: NodeRole
    """Raft role: ``VOTER``, ``STANDBY``, or ``SPARE``. Standby and
    spare nodes cannot become leader —
    ``ClusterClient.find_leader`` sorts voters first before probing."""


class NodeStore(ABC):
    """Abstract interface for storing cluster node information."""

    @abstractmethod
    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Return an immutable snapshot of known nodes.

        Implementations MUST return a Sequence that will not be mutated
        after return. A ``tuple`` is preferred (``MemoryNodeStore``
        uses one); a freshly-copied ``list`` is also acceptable as long
        as the implementation guarantees no subsequent mutation. Callers
        — notably ``ClusterClient.find_leader`` — iterate the returned
        value without defensive copies, and iteration is interleaved
        with ``await`` points. Returning a live backing collection that
        mutates under ``set_nodes()`` would produce torn reads during
        iteration.
        """
        ...

    @abstractmethod
    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes.

        Implementations MUST publish the update atomically from the
        perspective of ``get_nodes()``: a concurrent reader must see
        either the old snapshot or the new one, never a partially
        constructed sequence. ``MemoryNodeStore`` achieves this with a
        tuple-reference swap.
        """
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
                NodeInfo(node_id=i + 1, address=addr, role=NodeRole.VOTER)
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
