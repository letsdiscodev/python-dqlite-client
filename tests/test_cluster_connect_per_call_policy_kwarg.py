"""Pin: ``ClusterClient.connect`` accepts a per-call ``policy=`` kwarg
that overrides the instance-level redirect policy for this connect
attempt only, mirroring ``find_leader`` / ``cluster_info`` /
``leader_info``.
"""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def test_connect_accepts_policy_kwarg() -> None:
    sig = inspect.signature(ClusterClient.connect)
    assert "policy" in sig.parameters
    assert sig.parameters["policy"].kind == inspect.Parameter.KEYWORD_ONLY
    # Default is None so callers without a per-call override fall back
    # to the instance-level policy.
    assert sig.parameters["policy"].default is None
