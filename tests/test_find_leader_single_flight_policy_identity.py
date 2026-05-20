"""Pin: documented single-flight caveat re ``policy=`` identity.

The slot key on ``_find_leader_tasks`` is
``(trust_server_heartbeat, effective_policy)``. Lambdas / closures
/ partials compare by identity, so two callers passing fresh
``lambda a: True`` instances hash to different keys and run
independent sweeps. The collapse only fires when both callers
reach the same effective policy identity — typically by both
passing ``policy=None`` (resolves to the same
``self._redirect_policy`` identity).

The behaviour is intentional: per-call ``policy`` is an opt-OUT
of single-flight for audit-mode callers who need isolation.
Operators wanting both isolation AND collapse should construct
the ``ClusterClient`` with ``redirect_policy=<their policy>`` and
let callers pass ``policy=None``.

Pin: (1) the docstring explicitly documents the identity caveat
and the opt-out semantics; (2) the slot-key shape preserves the
collapse for the ``policy=None`` path.
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def test_find_leader_docstring_documents_policy_identity_caveat() -> None:
    """The docstring on ``find_leader`` must explain why per-call
    ``policy=`` identity defeats single-flight collapse, and how to
    opt-IN to collapse via constructor-time configuration."""
    doc = ClusterClient.find_leader.__doc__ or ""
    assert "policy callable identity" in doc, "docstring must explain the slot-key identity caveat"
    assert "redirect_policy" in doc, "docstring must point at the constructor-level alternative"
    assert "opt-OUT" in doc or "opt-out" in doc.lower(), (
        "docstring must label the per-call kwarg as opt-out semantics"
    )


def test_find_leader_slot_key_resolves_none_to_instance_default() -> None:
    """Inspection pin: the slot key resolves ``policy=None`` to
    ``self._redirect_policy`` BEFORE hashing, so two ``None`` callers
    deterministically share a slot. The key is
    ``(trust_server_heartbeat, effective_policy)``."""
    src = inspect.getsource(ClusterClient.find_leader)
    # The resolution step must precede the key construction.
    assert "policy if policy is not None else self._redirect_policy" in src
    # The key shape: tuple of (heartbeat-bool, effective-policy).
    assert "(trust_server_heartbeat, effective_policy)" in src
