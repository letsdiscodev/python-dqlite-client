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

    def __init__(
        self,
        addresses: list[str] | None = None,
        *,
        initial_addresses: list[str] | None = None,
    ) -> None:
        """Seed with a list of ``host:port`` addresses.

        ``addresses`` is the preferred name (matches the sibling
        ``ClusterClient.from_addresses`` / ``ConnectionPool`` /
        top-level ``connect`` APIs). ``initial_addresses`` is a
        deprecated alias kept for back-compat; prefer ``addresses`` in
        new code.
        """
        if addresses is not None and initial_addresses is not None:
            raise TypeError("Pass only one of 'addresses' or 'initial_addresses'")
        seed = addresses if addresses is not None else initial_addresses
        if seed:
            # Strip leading/trailing whitespace and reject empty
            # entries up front. Without this, a typoed seed like
            # ``["localhost:9001\n"]`` (e.g. read from a config
            # file) leaks ``ValueError`` through the sweep's narrow
            # exception filter and surfaces deep inside
            # ``find_leader``. Deduplicate while preserving order
            # so a config with the same address twice doesn't
            # double the probe count and double the per-node
            # error lines in the failure-aggregate message.
            seen: set[str] = set()
            unique: list[str] = []
            for raw in seed:
                if not isinstance(raw, str):
                    raise TypeError(
                        f"NodeStore addresses must be 'host:port' strings, got {type(raw).__name__}"
                    )
                addr = raw.strip()
                if not addr:
                    raise ValueError("NodeStore addresses must be non-empty 'host:port' strings")
                if addr in seen:
                    continue
                seen.add(addr)
                unique.append(addr)
            self._nodes: tuple[NodeInfo, ...] = tuple(
                NodeInfo(node_id=i + 1, address=addr, role=NodeRole.VOTER)
                for i, addr in enumerate(unique)
            )
        else:
            self._nodes = ()

    async def get_nodes(self) -> Sequence[NodeInfo]:
        """Get list of known nodes."""
        return self._nodes

    async def set_nodes(self, nodes: Sequence[NodeInfo]) -> None:
        """Update list of known nodes.

        Mirrors the strip / dedup / empty-rejection validation done
        in ``__init__``. Without this, a runtime update with a
        whitespace-laden or duplicated address would leak through
        and surface deep inside ``find_leader`` (whitespace) or
        inflate the probe count and per-node error lines
        (duplicates).
        """
        seen: set[str] = set()
        unique: list[NodeInfo] = []
        for node in nodes:
            if not isinstance(node.address, str):
                raise TypeError(
                    f"NodeInfo.address must be 'host:port' string, got "
                    f"{type(node.address).__name__}"
                )
            addr = node.address.strip()
            if not addr:
                raise ValueError("NodeInfo.address must be a non-empty 'host:port' string")
            if addr in seen:
                continue
            seen.add(addr)
            if addr is node.address:
                unique.append(node)
            else:
                # NodeInfo is a frozen dataclass; rebuild with the
                # canonical (stripped) address so a downstream lookup
                # by-address matches the canonical form.
                unique.append(NodeInfo(node_id=node.node_id, address=addr, role=node.role))
        self._nodes = tuple(unique)
