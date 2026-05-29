"""``_ProbeMiss.message`` on the stale-redirect arm sanitizes the server-supplied
``leader_address`` hint before it lands in ``ClusterError.args[0]``, so an
embedded LF cannot split downstream log records (CWE-117 log injection)."""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


@pytest.mark.asyncio
async def test_stale_redirect_probemiss_strips_lf_from_leader_hint() -> None:
    store = MemoryNodeStore(["node-a:9001"])
    cluster = ClusterClient(store, timeout=5.0)

    # Hint carries an embedded LF; the re-probe fails to confirm it (not in the
    # store), forcing the stale-redirect arm.
    malicious_hint = "evil.example:9001\nINJECTED LINE node-x:9001"

    call_count = 0

    async def fake_query_leader(address: str, **_kw: object) -> str | None:
        nonlocal call_count
        call_count += 1
        if address == "node-a:9001":
            return malicious_hint
        # Re-probe calls fail (None == "no leader known here").
        return None

    cluster._query_leader = fake_query_leader

    with pytest.raises(ClusterError) as ei:
        await cluster.find_leader()
    error_text = ei.value.args[0]
    assert "\n" not in error_text or "INJECTED LINE" not in error_text.split("\n")[1], (
        f"_ProbeMiss.message must strip LF from the server-supplied "
        f"leader hint so it cannot split downstream log records; got "
        f"text spanning lines: {error_text!r}"
    )
    # When present, the injected token must appear inline (escaped), never on a
    # new line an analyst could mistake for an independent log event.
    if "INJECTED LINE" in error_text:
        first_line = error_text.split("\n")[0]
        assert "INJECTED LINE" in first_line, (
            f"server-supplied LF must be stripped at emission so the "
            f"injected text stays inline with the rest of the message; "
            f"error_text was: {error_text!r}"
        )
