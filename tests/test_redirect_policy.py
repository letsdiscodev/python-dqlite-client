"""Leader-redirect allowlist policy: ``redirect_policy`` authorizes each redirect
target so a compromised peer can't steer the client to an arbitrary address."""

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
        """Empty allowlist rejects every redirect."""
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
        """The helper materializes the iterable once, so a generator is safe across calls."""
        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(x for x in ["10.0.0.1:9001", "10.0.0.2:9001"]),
        )
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.2:9001")):
            assert asyncio.run(cc.find_leader()) == "10.0.0.2:9001"
        # Second call still honors the allowlist, proving up-front materialization.
        with patch.object(cc, "_query_leader", new=AsyncMock(return_value="10.0.0.2:9001")):
            assert asyncio.run(cc.find_leader()) == "10.0.0.2:9001"

    def test_policy_rejection_short_circuits_connect_retry(self) -> None:
        """A deterministic redirect-policy rejection must not be retried."""
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

        # Without the retry exclusion this would be 3 (default max_attempts).
        assert call_count == 1, (
            f"Policy rejection must short-circuit retry; got {call_count} "
            f"probe attempts (expected 1)."
        )

    def test_policy_rejection_does_not_log_exhausted_attempts(self, caplog) -> None:
        """An unretried ClusterPolicyError must not emit the 'exhausted N attempts'
        WARNING, which would misreport one rejection as N transport failures."""
        import logging as _logging
        from unittest.mock import patch

        from dqliteclient.exceptions import ClusterPolicyError

        store = MemoryNodeStore(["10.0.0.1:9001"])
        cc = ClusterClient(
            store,
            timeout=5.0,
            redirect_policy=allowlist_policy(["10.0.0.1:9001"]),
        )

        async def probe(address: str, **kw: object) -> str | None:
            return "attacker.com:9001"

        caplog.set_level(_logging.WARNING, logger="dqliteclient.cluster")
        with (
            patch.object(cc, "_query_leader", side_effect=probe),
            pytest.raises(ClusterPolicyError),
        ):
            asyncio.run(cc.connect())

        exhausted = [
            r.getMessage()
            for r in caplog.records
            if r.name == "dqliteclient.cluster" and "exhausted" in r.getMessage()
        ]
        assert not exhausted, (
            f"connect() must not log 'exhausted attempts' for an unretried "
            f"ClusterPolicyError; got: {exhausted}"
        )

    def test_self_leader_does_not_bypass_policy(self) -> None:
        """Policy applies to every confirmed-leader address, including self-confirm.
        The seed list and policy are independent: an allowlist may exclude a seed."""
        from dqliteclient.exceptions import ClusterError, ClusterPolicyError

        store = MemoryNodeStore(["10.0.0.1:9001"])
        # Policy rejects everything, so the self-confirm path must propagate it.
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
        """Policy rejection emits a DEBUG log so an SSRF attempt is visible from logs alone."""
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
