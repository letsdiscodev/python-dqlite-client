"""Pin the ``_failure_message`` helper that renders FailureResponse
text for raised OperationalError / ProtocolError.

Two surface contracts:

1. Empty / whitespace-only server messages are substituted with
   ``"(no diagnostic from server)"`` so log scraping has a stable
   keyword to grep and operators are not staring at ``"[1] "``.
2. The address suffix is appended unchanged. Non-empty messages pass
   through verbatim (preserving the ``_sanitize_server_text`` cleanup
   already applied by the wire layer).
"""

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
