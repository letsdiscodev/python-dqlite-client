"""Pins for permanent documentation notes describing structural
divergences from go-dqlite. Each pin is a substring fence on a
docstring or in-source comment; a future refactor that drops the
note trips the pin so the reviewer keeps the rationale in place.

The pins cover four divergences:

1. ``ClusterClient.connect`` — cancel propagation through asyncio
   vs go's ``ctx.Done()`` check at iteration tops.
2. ``ClusterClient.connect`` and ``find_leader`` — double-dial
   structural cost (Python returns an address; Go adopts the
   probe socket).
3. ``DqliteProtocol.handshake`` — deliberate randomised
   ``client_id`` divergence from go-dqlite's default-zero.
4. ``DqliteProtocol._operation_deadline`` — per-phase timeout
   vs Go's absolute per-call ``conn.SetDeadline`` deadline.
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient
from dqliteclient.protocol import DqliteProtocol


def test_cluster_connect_documents_cancel_propagation_divergence() -> None:
    doc = ClusterClient.connect.__doc__ or ""
    assert "Comparison to go-dqlite" in doc
    assert "Cancel-propagation discipline" in doc
    # The "first attempt always runs" Go behaviour is the load-bearing
    # contrast point.
    assert "first attempt" in doc.lower()


def test_cluster_connect_documents_double_dial_cost() -> None:
    doc = ClusterClient.connect.__doc__ or ""
    assert "double-dial" in doc.lower() or "extra TCP three-way" in doc
    # Operators reading this note must understand they pay one extra
    # RTT per successful connect on the leader.
    assert "extra" in doc.lower() and "round trip" in doc.lower()


def test_find_leader_documents_address_only_return() -> None:
    """``find_leader``'s docstring documents that it returns an
    address (not an open protocol), so high-level callers re-dial."""
    doc = ClusterClient.find_leader.__doc__ or ""
    assert "address" in doc.lower()
    assert "re-dial" in doc.lower() or "extra TCP three-way" in doc


def test_handshake_documents_client_id_divergence_rationale() -> None:
    """The source comment at the randomisation site documents the
    Go-divergence rationale: Go is default-zero, Python randomises
    for log distinguishability, the cost / fork-safety trade-off
    against ``random.getrandbits`` lives in the comment."""
    src = inspect.getsource(DqliteProtocol.handshake)
    assert "Deliberate divergence from go-dqlite" in src
    # The operational trade-offs the comment must call out:
    assert "Mixed-client clusters" in src
    assert "secrets.randbits" in src
    # Cross-references to sibling SystemRandom sites for symmetry.
    assert "_retry_random" in src or "_cluster_random" in src


def test_operation_deadline_documents_per_phase_worst_case_quantified() -> None:
    """The ``_operation_deadline`` docstring documents the per-phase
    vs Go-absolute-deadline divergence AND the quantitative
    worst-case multiplier."""
    src = inspect.getsource(DqliteProtocol._operation_deadline)
    # Quantitative example (the 10-frame / 120s shape).
    assert "10-frame" in src or "10 × continuation" in src.replace("times", "×")
    assert "120" in src
    # The trust_server_heartbeat amplification call-out.
    assert "300 s" in src or "300s" in src
    # The Go contrast remains.
    assert "differs from go-dqlite" in src
