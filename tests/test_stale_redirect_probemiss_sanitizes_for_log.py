"""Pin: ``_ProbeMiss.message`` at the stale-redirect arm strips LF/CR
before it lands in ``ClusterError.args[0]``.

The stale-redirect arm in ``find_leader`` composes a ``_ProbeMiss``
whose ``message`` field carries the responder address plus the
server-supplied ``leader_address`` hint. ``_ProbeMiss.message`` ends
up in ``ClusterError.args[0]`` (and from there into downstream
``logger.exception(...)`` records). A server-controlled
``leader_address`` containing an embedded ``\\n`` would split the log
record across lines (CWE-117 log injection, secondary surface).

Mirrors the round-1 fix at the sibling probe-failure arm (done issue
``probe-miss-message-str-e-no-lf-strip``): apply ``sanitize_for_log``
at the emission site so every downstream log path inherits the
LF-stripped form.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_stale_redirect_probemiss_strips_lf_from_leader_hint() -> None:
    """A peer returns a redirect with an LF embedded in
    ``leader_address``. The verify-redirect re-probe fails (the hint
    target isn't in the store, so it fails to connect). The
    ``_ProbeMiss.message`` text must NOT contain a raw LF — it
    should be LF-escaped via ``sanitize_for_log``.
    """
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    # Node A redirects to a leader-hint that contains an embedded LF.
    # The verify-redirect re-probe will fail to confirm (the hint
    # target is not in the store), forcing the stale-redirect arm.
    malicious_hint = "evil.example:9001\nINJECTED LINE node-x:9001"

    call_count = 0

    async def fake_query_leader(address: str, **_kw: object) -> str | None:
        nonlocal call_count
        call_count += 1
        if address == "node-a:9001":
            # First call: legitimate sweep — return a malicious hint.
            # Verify-redirect re-probe runs against the hint address;
            # we want that re-probe to fail so the stale-redirect arm
            # fires.
            return malicious_hint
        # All re-probe calls fail (None == "no leader known here").
        return None

    cluster._query_leader = fake_query_leader

    # The sweep exhausts and raises ClusterError. The aggregated text
    # must not carry a raw LF — that would be the CWE-117 surface.
    with pytest.raises(ClusterError) as ei:
        await cluster.find_leader()
    error_text = ei.value.args[0]
    assert "\n" not in error_text or "INJECTED LINE" not in error_text.split("\n")[1], (
        f"_ProbeMiss.message must strip LF from the server-supplied "
        f"leader hint so it cannot split downstream log records; got "
        f"text spanning lines: {error_text!r}"
    )
    # Stronger: the hint text appears in the message in escaped form
    # (sanitize_for_log replaces \n with the literal escape sequence
    # or strips it). The injected token therefore appears inline,
    # never on a new line, when present.
    if "INJECTED LINE" in error_text:
        first_line = error_text.split("\n")[0]
        # The injected text must live on the SAME logical line as
        # the rest of the stale-redirect record — not on a separate
        # log line that an analyst would mistake for an independent
        # event.
        assert "INJECTED LINE" in first_line, (
            f"server-supplied LF must be stripped at emission so the "
            f"injected text stays inline with the rest of the message; "
            f"error_text was: {error_text!r}"
        )
