"""Pin: ``ClusterClient.dump()`` docstring must NOT assert that the
upstream protocol requires the request to land on the leader. Sending
to the leader is a Python implementation choice — ``gateway.c::
handle_dump`` does NOT call ``CHECK_LEADER``, so any cluster member
(voter, standby, spare) can serve the dump. A reader looking to back
up from a spare deserves the hedge that ``cluster_info`` already
provides ("any node can answer this ... but this method asks the
leader so ...").

Mirrors ``cluster_info``'s docstring shape — see
``src/dqliteclient/cluster.py`` for the ``cluster_info`` hedge
("Any node can answer this — Raft replicates the configuration —
but this method asks the leader so the returned view is the freshest
one available").
"""

from __future__ import annotations

from dqliteclient.cluster import ClusterClient


def test_dump_docstring_does_not_overclaim_leader_only() -> None:
    """The dump docstring must hedge the leader-only claim. Either
    drop the "sent to the leader" wording or surround it with a
    "Python design choice / upstream does not require it" hedge.
    """
    doc = ClusterClient.dump.__doc__ or ""
    if "sent to the leader" in doc:
        # If the sentence is preserved, it MUST be hedged with one
        # of the recognised disclaimers so the reader knows the
        # leader requirement is not protocol-level.
        hedge_markers = (
            "design choice",
            "any cluster member",
            "any node",
            "does not require",
            "does not call",
        )
        lower = doc.lower()
        assert any(marker in lower for marker in hedge_markers), (
            "dump() docstring asserts 'sent to the leader' without "
            "hedging; gateway.c::handle_dump does NOT call CHECK_LEADER, "
            "so the docstring must clarify that the leader routing is "
            "a Python design choice rather than a protocol requirement. "
            "Mirrors the cluster_info docstring's discipline."
        )
