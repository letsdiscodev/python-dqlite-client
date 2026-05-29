"""``_failure_message`` renders FailureResponse text: empty/whitespace
messages become "(no diagnostic from server)"; the address suffix is
appended unchanged and non-empty bodies pass through verbatim."""

from __future__ import annotations

import pytest

from dqliteclient.protocol import _failure_message


class TestFailureMessage:
    @pytest.mark.parametrize(
        ("message", "addr_suffix", "expected"),
        [
            ("real msg", " to host:9001", "real msg to host:9001"),
            ("real msg", "", "real msg"),
            ("", " to host:9001", "(no diagnostic from server) to host:9001"),
            ("", "", "(no diagnostic from server)"),
            ("   ", " to host:9001", "(no diagnostic from server) to host:9001"),
            ("\t\n", "", "(no diagnostic from server)"),
        ],
    )
    def test_renders_message_with_address(
        self, message: str, addr_suffix: str, expected: str
    ) -> None:
        assert _failure_message(message, addr_suffix) == expected
