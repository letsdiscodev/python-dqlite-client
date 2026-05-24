"""Leader-redirect allowlist policy.

A compromised peer can return any address as the leader; the client
used to open a TCP connection to whatever it was told. ClusterClient
now takes an optional ``redirect_policy`` callable that authorizes
each redirect target; a convenience ``allowlist_policy`` helper builds
one from a static list of addresses.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dqliteclient.cluster import ClusterClient, allowlist_policy
from dqliteclient.exceptions import ClusterError
from dqliteclient.node_store import MemoryNodeStore


class TestRedirectPolicy:
    def test_accepts_seeded_redirect(self) -> None:
        store = MemoryNodeStore(["10.0.0.1:9001", "10.0.0.2:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(["10.0.0.1:9001", "10.0.0.2:9001"]),
        )
        # Simulate: probing 10.0.0.1 reports 10.0.0.2 as leader.
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.2:9001")):
            result = asyncio.run(cc.find_leader())
        assert result == "10.0.0.2:9001"

    def test_rejects_unknown_redirect(self) -> None:
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
        )
        # A compromised peer redirects to an attacker-controlled host.
        with (
            patch.object(cc, "_query_leader", new=AsyncMock(return_value="attacker.com:9001")),
            pytest.raises(ClusterError, match="rejected"),
        ):
            asyncio.run(cc.find_leader())

    def test_no_policy_means_any_redirect_accepted(self) -> None:
        """Default (None) policy preserves legacy behavior."""
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(store, timeout=5.0)
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="anywhere.invalid:9001")):
            result = asyncio.run(cc.find_leader())
        assert result == "anywhere.invalid:9001"

    def test_empty_allowlist_rejects_all_redirects(self) -> None:
        """Empty allowlist means every redirect fails; self-leader still
        works because that path bypasses the policy entirely (it's not a
        real redirect)."""
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy([]),
        )
        with (
            patch.object(cc, "_query_leader", new=AsyncMock(return_value="other:9001")),
            pytest.raises(ClusterError, match="rejected"),
        ):
            asyncio.run(cc.find_leader())

    def test_allowlist_accepts_iterator_input(self) -> None:
        """The helper accepts any iterable; internally it materializes once
        into a set, so a generator is safe (no iterator-exhaustion trap on
        repeated calls)."""
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(x for x in ["10.0.0.1:9001", "10.0.0.2:9001"]),
        )
        # First call — would have drained the generator already.
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.2:9001")):
            assert asyncio.run(cc.find_leader()) == "10.0.0.2:9001"
        # Second call still honors the allowlist (proves the set was
        # materialized up-front, not re-iterated).
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.2:9001")):
            assert asyncio.run(cc.find_leader()) == "10.0.0.2:9001"

    def test_policy_rejection_short_circuits_connect_retry(self) -> None:
        """A deterministic redirect-policy rejection must not be
        retried. Before the ClusterPolicyError split it was caught as
        a plain ClusterError by retry_with_backoff's retryable tuple
        and re-attempted ``max_attempts`` times, multiplying the
        wall-clock cost with no chance of recovery.
        """
        from unittest.mock import patch

        from dqliteclient.exceptions import ClusterPolicyError

        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
        )

        call_count = 0

        async def probe(address: str, **kw: object) -> str | None:
            nonlocal call_count
            call_count += 1
            return "attacker.com:9001"

        with (
            patch.object(cc, "_query_leader", side_effect=probe),
            pytest.raises(ClusterPolicyError, match="rejected"),
        ):
            asyncio.run(cc.connect())

        # One find_leader call → one rejection. Without the exclusion,
        # retry_with_backoff (max_attempts=3 by default) would invoke
        # find_leader three times.
        assert call_count == 1, (
            f"Policy rejection must short-circuit retry; got {call_count} "
            f"probe attempts (expected 1)."
        )

    def test_self_leader_does_not_bypass_policy(self) -> None:
        """Policy applies to EVERY confirmed-leader address, including
        the self-confirm case where the probed node returns its own
        address. The seed list and the policy are independent — an
        allowlist policy can legitimately exclude addresses present in
        the seed list (the canonical regional-pin use case). Pre-
        existing behavior silently bypassed the policy on self-confirm;
        the fix lifts the policy check out of the inner
        ``if not _addr_equiv`` so it runs on every leader hit. See
        ``test_probe_one_redirect_policy_on_self_confirm.py`` for the
        full pin set covering both arms."""
        from dqliteclient.exceptions import ClusterError, ClusterPolicyError

        store = MemoryNodeStore(["10.0.0.1:9001"])
        # Policy rejects everything — the self-confirm path must
        # propagate the rejection, not silently return the address.
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=lambda _a: False,
        )
        with (
            patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.1:9001")),
            pytest.raises((ClusterPolicyError, ClusterError)),
        ):
            asyncio.run(cc.find_leader())

    def test_redirect_rejection_emits_debug_log(self, caplog) -> None:
        """Policy rejection must emit a DEBUG log so an SSRF-style
        attempt or a policy misconfiguration is visible from logs
        alone — not only through the exception stack."""
        import logging as _logging

        from dqliteclient.exceptions import ClusterPolicyError

        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=lambda _a: False,
        )

        caplog.set_level(_logging.DEBUG, logger="dqliteclient.cluster")
        with pytest.raises(ClusterPolicyError):
            cc._check_redirect("attacker.com:9001")

        messages = [r.getMessage() for r in caplog.records if r.name == "dqliteclient.cluster"]
        assert any(
            "redirect rejected by policy" in m and "attacker.com:9001" in m for m in messages
        )
