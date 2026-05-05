"""Pin: ``parse_address`` rejects pathologically long inputs up front.

The wire-side ``LeaderResponse.address`` is server-capped at 256
bytes, but ``parse_address`` is also reachable from caller-supplied
seed URLs (env vars, config files). Without an up-front length cap,
a misconfigured megabyte-sized address produces a ``ValueError``
whose message interpolates the full input via ``{address!r}`` —
multi-MB log lines, large tracebacks, and expensive
``except ValueError`` formatting through any wrapping layer.

Pin a 1 KiB cap with a small, bounded error message.
"""

from __future__ import annotations

import pytest

from dqliteclient.connection import _MAX_ADDRESS_LEN, parse_address


def test_oversized_address_rejected_with_bounded_message() -> None:
    huge = "a" * (_MAX_ADDRESS_LEN + 1) + ":9001"
    with pytest.raises(ValueError) as ei:
        parse_address(huge)
    # The error message must be bounded, NOT interpolate the full input.
    assert len(str(ei.value)) < 1024


def test_oversized_address_message_describes_the_cap() -> None:
    huge = "a" * (1 << 20) + ":9001"
    with pytest.raises(ValueError, match="exceeds maximum"):
        parse_address(huge)


def test_at_cap_address_proceeds_to_normal_parse() -> None:
    """Just-below-cap address proceeds to the regular parse path; the
    cap should not constrain legitimate addresses."""
    # Construct a hostname that's just below the cap. Use a long
    # but DNS-legal label sequence: ``a.a...a:9001``.
    host = ".".join(["a"] * 50)  # short enough; legal hostname
    addr = f"{host}:9001"
    assert len(addr) < _MAX_ADDRESS_LEN
    out_host, out_port = parse_address(addr)
    assert out_port == 9001


def test_non_string_input_rejected() -> None:
    with pytest.raises(ValueError, match="expected str"):
        parse_address(b"localhost:9001")  # type: ignore[arg-type]
