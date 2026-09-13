"""Argument validation shared by every entry point: addresses, timeouts, defaults."""

import ipaddress
import math
import os
import re
from typing import Final

from dqlitewire.messages.responses import _MAX_ADDRESS_SIZE as _WIRE_MAX_ADDRESS_SIZE

__all__ = [
    "validate_max_attempts",
    "CLOSE_TIMEOUT_FLOOR",
    "CLOSE_TIMEOUT_FLOOR_RATIONALE",
    "DEFAULT_CLOSE_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "get_current_pid",
    "parse_address",
    "validate_timeout",
]


def get_current_pid() -> int:
    return os.getpid()


# RFC 1035 hostname labels, dotted sequence up to 253 chars. A single
# trailing dot (root-anchored FQDN) is accepted and dropped in canonical
# form so the two surface variants compare equal for allowlists.
_HOSTNAME_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?=.{1,254}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)"
    r"(?:\.(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?))*"
    r"\.?$"
)


def _canonicalize_host(host: str, address: str) -> str:
    """Validate and canonicalize a host (IPv4/IPv6 literal or ASCII hostname).

    Rejects credentials-like '@', whitespace/CRLF, and non-ASCII (IDN)
    hosts so a server-controlled redirect cannot smuggle log-injection or
    DNS-rebinding vectors past the parser.
    """
    if not host:
        raise ValueError(f"Invalid address format: empty hostname in {address!r}")
    # IPv6 shorthand (``::1``) must canonicalize so allowlists see one form.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        # Reject IP literals that cannot legitimately be a TCP destination
        # (unspecified / multicast / reserved): they pass parsing but fail
        # at connect, and an allowlist containing the unspecified IP would
        # silently authorise every redirect there.
        # Unwrap IPv4-mapped IPv6 (``::ffff:0.0.0.0``) so the embedded
        # IPv4's classification governs — ``is_unspecified`` on the wrapper
        # is False even when the embedded IPv4 is unspecified.
        ipv4_mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
        effective: ipaddress.IPv4Address | ipaddress.IPv6Address = (
            ipv4_mapped if ipv4_mapped is not None else ip
        )
        if effective.is_unspecified:
            raise ValueError(
                f"Invalid host in address {address!r}: "
                f"{host!r} is the unspecified IP literal and not a valid TCP destination"
            )
        if effective.is_multicast:
            raise ValueError(
                f"Invalid host in address {address!r}: "
                f"{host!r} is a multicast IP and not a valid TCP destination"
            )
        # Reject ``is_reserved`` only for IPv4 (240.0.0.0/4 Class-E): CPython
        # classifies IPv6 loopback ``::1`` as reserved, and rejecting it
        # would break local-test harnesses using ``[::1]:port``.
        if isinstance(effective, ipaddress.IPv4Address) and effective.is_reserved:
            raise ValueError(
                f"Invalid host in address {address!r}: "
                f"{host!r} is a reserved IP and not a valid TCP destination"
            )
        # IPv4-mapped IPv6 canonicalises to the embedded IPv4 dotted-quad
        # (RFC 4291 §2.5.5.2) so an allowlist of ``127.0.0.1`` matches a
        # redirect to ``[::ffff:127.0.0.1]``.
        return str(effective)
    # Reject IDN outright: punycode does not round-trip reliably on the
    # wire and non-ASCII hostnames are a homograph-attack vector.
    try:
        host.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(
            f"Invalid host in address {address!r}: non-ASCII hostnames are not supported"
        ) from e
    if not _HOSTNAME_LABEL_RE.match(host):
        raise ValueError(
            f"Invalid host in address {address!r}: {host!r} is not a valid hostname or IP literal"
        )
    # Strip the trailing FQDN dot so rooted and unrooted forms compare equal.
    return host.rstrip(".").lower()


def validate_timeout(
    value: float,
    *,
    name: str = "timeout",
    min_value: float = 0.0,
    min_value_rationale: str | None = None,
) -> float:
    """Validate a user-supplied timeout: positive, finite, not ``bool``.

    ``bool`` is rejected explicitly (``True`` would otherwise pass as
    ``1.0``); ``inf`` / ``nan`` fail here rather than later inside
    ``asyncio.wait_for``. ``min_value`` (exclusive, default ``0.0``) is
    the floor; ``min_value_rationale`` is appended to the diagnostic so a
    caller can supply its own explanation.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r} (bool)")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    if value < min_value:
        msg = f"{name} must be >= {min_value}; got {value}"
        if min_value_rationale:
            msg += f". {min_value_rationale}"
        raise ValueError(msg)
    return float(value)


# Floor + rationale shared by every ``close_timeout`` validator caller (this
# module, pool, dbapi, SA URL validator) so the value cannot drift.
CLOSE_TIMEOUT_FLOOR_RATIONALE: Final[str] = (
    "Below this floor, the dispose-time writer-close may complete before "
    "FIN flushes, leaving connections lingering in TIME_WAIT."
)
CLOSE_TIMEOUT_FLOOR: Final[float] = 0.01

# Operator-visible timeout defaults, shared across every entry point so a
# tuning lands in lockstep. ``timeout`` is per-RPC-phase (see
# DqliteProtocol._operation_deadline); ``close_timeout`` is sized for LAN
# FIN/ACK — bump for WAN, shrink for SIGTERM-bound deployments.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_CLOSE_TIMEOUT_SECONDS: Final[float] = 0.5


# Shared with the wire layer's address-decode cap: a seed above this could
# never round-trip through cluster discovery or redirect.

_MAX_ADDRESS_LEN: Final[int] = _WIRE_MAX_ADDRESS_SIZE


def parse_address(address: str) -> tuple[str, int]:
    """Parse a host:port address into ``(canonical_host, port)``.

    IP literals are canonicalized; hostnames lowercased. Invalid hosts
    (credentials-like '@', whitespace/CRLF, non-ASCII, empty) raise
    ``ValueError``. Stable public surface; the ``_parse_address`` alias
    is kept for backwards compatibility.
    """
    # Length cap first: a misconfigured megabyte-sized seed would otherwise
    # interpolate the full input via ``{address!r}`` into a multi-MB error.
    if not isinstance(address, str):
        raise ValueError(f"Invalid address: expected str, got {type(address).__name__}")
    if len(address) > _MAX_ADDRESS_LEN:
        raise ValueError(
            f"Invalid address: length {len(address)} exceeds maximum {_MAX_ADDRESS_LEN}"
        )
    # Reject embedded NUL early with a specific diagnostic (downstream
    # guards conflate it with generic shape failures).
    if "\x00" in address:
        raise ValueError(f"Invalid address: contains NUL byte at offset {address.index(chr(0))}")

    if address.startswith("["):
        # Bracketed IPv6: [host]:port. RFC 3986 reserves brackets for
        # IPv6 literals; bracketed IPv4 / hostname / empty are rejected.
        if "]:" not in address:
            raise ValueError(
                f"Invalid IPv6 address format: expected '[host]:port', got {address!r}"
            )
        bracket_end = address.index("]")
        host = address[1:bracket_end]
        port_str = address[bracket_end + 2 :]  # Skip ']:'

        # RFC 6874: percent-decode the zone-ID suffix so the URI form
        # ``[fe80::1%25eth0]`` and the app form ``[fe80::1%eth0]`` match.
        from urllib.parse import unquote

        zone_sep = host.find("%")
        if zone_sep != -1:
            host = host[:zone_sep] + unquote(host[zone_sep:])
            # Reject pathological zone shapes here for a specific
            # diagnostic instead of the generic regex-fallback message.
            zone = host[zone_sep + 1 :]
            if not zone:
                raise ValueError(
                    f"Bracket syntax in {address!r} has an empty IPv6 zone "
                    f"identifier (after '%'); supply a zone like '%eth0' "
                    f"or remove the '%'"
                )
            if any(c.isspace() or c == "/" for c in zone):
                raise ValueError(
                    f"Bracket syntax in {address!r} has an invalid IPv6 "
                    f"zone identifier {zone!r}; zone IDs must not contain "
                    f"whitespace or '/'"
                )

        # ``ipaddress.ip_address`` rejects the ``%zone`` suffix; strip it.
        ipv6_part = host.split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(ipv6_part)
        except ValueError as e:
            raise ValueError(
                f"Bracket syntax in {address!r} is reserved for IPv6 "
                f"literals; {host!r} is not an IPv6 address"
            ) from e
        if not isinstance(parsed, ipaddress.IPv6Address):
            raise ValueError(
                f"Bracket syntax in {address!r} is reserved for IPv6 "
                f"literals; got {type(parsed).__name__}"
            )
    else:
        if ":" not in address:
            raise ValueError(f"Invalid address format: expected 'host:port', got {address!r}")
        host, port_str = address.rsplit(":", 1)
        # Diagnose unbracketed IPv6 before the port parse so ``"::1:abc"``
        # reports missing brackets, not "invalid port". ``@`` is left for
        # ``_canonicalize_host`` (credentials-smuggle, more specific msg).
        if ":" in host and "@" not in host:
            raise ValueError(
                f"IPv6 addresses must be bracketed: got {address!r}, expected '[host]:port'"
            )

    # Strict port parse: stdlib ``int()`` accepts whitespace, unary ``+``,
    # underscores, and Unicode digits, which would break allowlist matching
    # against a peer redirect. Restrict to plain ASCII digits; allow a
    # leading ``-`` so negatives hit the "not in range" diagnostic.
    if port_str.startswith("-") and port_str[1:].isascii() and port_str[1:].isdigit():
        port = int(port_str)  # negative — fails the range check below
    elif port_str.isascii() and port_str.isdigit():
        port = int(port_str)
    else:
        raise ValueError(f"Invalid port in address {address!r}: {port_str!r} is not a number")

    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port in address {address!r}: {port} is not in range 1-65535")

    host = _canonicalize_host(host, address)
    return host, port


# Backwards-compatible alias for the leading-underscore name.


def validate_max_attempts(value: int | None) -> int | None:
    """``None`` (use the default) or an int >= 1; bools are rejected."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"max_attempts must be int or None, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"max_attempts must be at least 1, got {value}")
    return value
