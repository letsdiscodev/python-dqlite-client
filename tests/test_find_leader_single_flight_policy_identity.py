"""The single-flight slot key is keyed by policy identity, so per-call
``policy=`` lambdas opt out of collapse while ``policy=None`` callers still
share a slot. Pins the documented caveat and the slot-key shape."""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def test_find_leader_slot_key_resolves_none_to_instance_default() -> None:
    """Inspection pin: the slot key resolves ``policy=None`` to
    ``self._redirect_policy`` before hashing so two None callers share a
    slot."""
    src = inspect.getsource(ClusterClient.find_leader)
    assert "policy if policy is not None else self._redirect_policy" in src
    assert "(trust_server_heartbeat, effective_policy)" in src
