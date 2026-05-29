"""connect accepts a per-call policy= kwarg that overrides the instance-level
redirect policy for this attempt only."""

from __future__ import annotations

import inspect

from dqliteclient.cluster import ClusterClient


def test_connect_accepts_policy_kwarg() -> None:
    sig = inspect.signature(ClusterClient.connect)
    assert "policy" in sig.parameters
    assert sig.parameters["policy"].kind == inspect.Parameter.KEYWORD_ONLY
    # Default None so callers without an override fall back to the instance policy.
    assert sig.parameters["policy"].default is None
