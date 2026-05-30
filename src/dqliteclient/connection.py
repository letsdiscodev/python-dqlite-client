"""High-level connection interface for dqlite."""

import asyncio
import contextlib
import ipaddress
import logging
import math
import os
import re
import string
import warnings
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Final, NoReturn, Self

from dqliteclient._dial import (
    DialFunc,
    open_connection,
)
from dqliteclient.exceptions import (
    AmbiguousCommitError,
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.protocol import (
    DqliteProtocol,
    validate_positive_int_or_none,
)
from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES
from dqlitewire import DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS
from dqlitewire import (
    LEADER_ERROR_CODES,
    LEADER_LOST_DB_LOOKUP_SUBSTRING,
    NO_TRANSACTION_MESSAGE_SUBSTRINGS,
    WIRE_DECODE_FAILED_PREFIX,
    sanitize_for_log,
)
from dqlitewire import SQLITE_BUSY as _SQLITE_BUSY
from dqlitewire import SQLITE_NOTFOUND as _SQLITE_NOTFOUND
from dqlitewire import TX_AUTO_ROLLBACK_PRIMARY_CODES as _TX_AUTO_ROLLBACK_PRIMARY_CODES
from dqlitewire import EncodeError as _WireEncodeError
from dqlitewire import primary_sqlite_code as _primary_sqlite_code
from dqlitewire import sanitize_server_text as _sanitize_display_text

__all__ = ["DqliteConnection"]

logger = logging.getLogger(__name__)


def get_current_pid() -> int:
    """Return the current process pid, syscall-direct (no cache).

    A cached value would lag a ``fork()`` in the child until our
    ``register_at_fork`` callback ran, after any third-party callback
    that races us — a fork guard miss is silent wire corruption.
    """
    return os.getpid()


# Re-snapshot cap for ``_pending_drain`` retire arms in ``_connect_impl`` /
# ``_close_impl``. Shared with :mod:`dqliteclient.pool` so its per-iteration
# drain cap stays in lockstep; worst-case inner close path is
# ``(_CLOSE_RESNAPSHOT_CAP + 1) × close_timeout``.
_CLOSE_RESNAPSHOT_CAP: Final[int] = 3

# Row stride at which ``fetch`` cedes the event loop while reshaping a decoded
# result into dicts, so a large un-yielding reshape does not freeze the loop.
_FETCH_DICT_YIELD_EVERY: Final[int] = 4096

# Bare ``BEGIN`` (implicit DEFERRED) matches the C/Go peer clients; dqlite's
# Raft FSM is SERIALIZABLE regardless of qualifier. Literals pinned so a
# refactor can't silently switch to a vendor-qualified form — and so
# ``transaction()`` and the pool reset-on-return path stay consistent (a
# divergence could return a connection with an open server-side tx).
_TRANSACTION_BEGIN_SQL: Final[str] = "BEGIN"
_TRANSACTION_COMMIT_SQL: Final[str] = "COMMIT"
_TRANSACTION_ROLLBACK_SQL: Final[str] = "ROLLBACK"

# ``_TX_AUTO_ROLLBACK_PRIMARY_CODES`` (imported from ``dqlitewire``): primary
# SQLite codes for which upstream ``leader.c`` has already cleared the
# server-side tx (it polls ``sqlite3_txn_state`` and clears on
# ``SQLITE_TXN_NONE``). On these the connection is healthy but the local
# ``_in_transaction`` / ``_tx_owner`` must be cleared, else the next
# statement silently auto-begins a fresh tx the user doesn't know about.


_BARE_IDENT_FIRST: Final[frozenset[str]] = frozenset(string.ascii_letters + "_")
_BARE_IDENT_REST: Final[frozenset[str]] = frozenset(string.ascii_letters + string.digits + "_")

# Message fragments for the Raft-side BUSY path ("in-flight write not
# accepted; server tx may be gone"), coupled to gateway.c failure() wording.
# Other Raft-BUSY paths are indistinguishable from engine-BUSY at this layer.
_RAFT_BUSY_MESSAGE_FRAGMENTS: Final[tuple[str, ...]] = ("checkpoint in progress",)


def _is_keyword_boundary(s: str, kw_len: int) -> bool:
    """True if position ``kw_len`` in ``s`` ends an SQL keyword.

    Not ``str.isalnum``: SQLite treats ``_`` as an identifier char, so a
    boundary check using isalnum would split ``SAVEPOINT_foo`` mid-token.
    """
    return len(s) == kw_len or s[kw_len] not in _BARE_IDENT_REST


def _split_top_level_statements(sql: str) -> list[str]:
    """Split SQL on top-level ``;`` boundaries, returning stripped pieces.

    The dqlite EXEC path runs multi-statement input, but the tx tracker's
    prefix-sniff sees only the leading verb, so splitting lets it
    re-classify each piece. Skips ``;`` inside string/identifier literals
    and comments. Inside a ``CREATE [TEMP] TRIGGER ... BEGIN ... END``
    body, ``;`` is an inner terminator and must not split the outer DDL.
    """
    out: list[str] = []
    start = 0
    i = 0
    n = len(sql)
    # ``case_depth`` lets an inner ``CASE WHEN ... END`` close before its
    # ``END`` decrements ``trigger_depth`` (BEGIN..END nesting depth).
    in_trigger_body = False
    trigger_depth = 0
    case_depth = 0
    trigger_scan_start = 0
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == '"':
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "[":
            i += 1
            while i < n and sql[i] != "]":
                i += 1
            if i < n:
                i += 1
            continue
        if c == "`":
            i += 1
            while i < n:
                if sql[i] == "`":
                    if i + 1 < n and sql[i + 1] == "`":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c.isalpha() and (i == 0 or not _is_word_char(sql[i - 1])):
            kw_end = i
            while kw_end < n and _is_word_char(sql[kw_end]):
                kw_end += 1
            kw = sql[i:kw_end].upper()
            if not in_trigger_body:
                if kw == "CREATE" and i >= trigger_scan_start:
                    j = _scan_for_trigger_begin(sql, kw_end, n)
                    if j > 0:
                        in_trigger_body = True
                        trigger_depth = 1
                        i = j
                        continue
            else:
                if kw == "BEGIN":
                    trigger_depth += 1
                    i = kw_end
                    continue
                if kw == "CASE":
                    case_depth += 1
                    i = kw_end
                    continue
                if kw == "END":
                    if case_depth > 0:
                        case_depth -= 1
                    else:
                        trigger_depth -= 1
                        if trigger_depth == 0:
                            in_trigger_body = False
                    i = kw_end
                    continue
            i = kw_end
            continue
        if c == ";" and not in_trigger_body:
            piece = sql[start:i].strip()
            if piece:
                out.append(piece)
            start = i + 1
            trigger_scan_start = start
        i += 1
    tail = sql[start:].strip()
    if tail:
        out.append(tail)
    return out


def _is_word_char(c: str) -> bool:
    """True if ``c`` is part of a SQL keyword/identifier word."""
    return c.isalnum() or c == "_"


def _skip_ws_and_comments(sql: str, i: int, n: int) -> int:
    """Advance past whitespace and SQL comments (``--`` line, ``/* */`` block)."""
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        break
    return i


def _scan_for_trigger_begin(sql: str, after_create: int, n: int) -> int:
    """Look ahead from just after ``CREATE`` for ``[TEMP|TEMPORARY] TRIGGER
    ... BEGIN``; return the index past ``BEGIN`` on success, else 0.

    Skips quoted identifiers, comments, and parenthesised sub-expressions
    (the ``WHEN (...)`` clause). Stops at any ``;`` or end-of-input.
    """
    i = after_create
    i = _skip_ws_and_comments(sql, i, n)
    if i >= n:
        return 0
    j = i
    while j < n and _is_word_char(sql[j]):
        j += 1
    word = sql[i:j].upper()
    if word in ("TEMP", "TEMPORARY"):
        i = _skip_ws_and_comments(sql, j, n)
        j = i
        while j < n and _is_word_char(sql[j]):
            j += 1
        word = sql[i:j].upper()
    if word != "TRIGGER":
        return 0
    i = j
    # Scan for the next standalone BEGIN at the same nesting level.
    paren_depth = 0
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == '"':
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "[":
            i += 1
            while i < n and sql[i] != "]":
                i += 1
            if i < n:
                i += 1
            continue
        if c == "`":
            i += 1
            while i < n:
                if sql[i] == "`":
                    if i + 1 < n and sql[i + 1] == "`":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == "(":
            paren_depth += 1
            i += 1
            continue
        if c == ")":
            if paren_depth > 0:
                paren_depth -= 1
            i += 1
            continue
        if c == ";":
            # CREATE TRIGGER ended without a BEGIN (short-form trigger).
            return 0
        if paren_depth == 0 and c.isalpha() and (i == 0 or not _is_word_char(sql[i - 1])):
            j = i
            while j < n and _is_word_char(sql[j]):
                j += 1
            if sql[i:j].upper() == "BEGIN":
                return j
            i = j
            continue
        i += 1
    return 0


# Longest-first ordered tuple (not a frozenset): ``startswith`` must match
# the longest candidate first so a future prefix-sharing verb is safe.
_TX_CONTROL_VERBS: Final[tuple[str, ...]] = (
    "SAVEPOINT",
    "ROLLBACK",
    "RELEASE",
    "COMMIT",
    "BEGIN",
    "END",
)


def _starts_with_tx_verb(stmt: str) -> bool:
    """True if ``stmt`` starts with a transaction-control verb.

    Strips leading comments and respects keyword boundaries so
    ``BEGIN_foo`` (an identifier) is not mistaken for the BEGIN keyword.
    """
    s = _strip_leading_comments(stmt)
    if not s:
        return False
    upper = s.upper()
    for verb in _TX_CONTROL_VERBS:
        if upper.startswith(verb) and _is_keyword_boundary(upper, len(verb)):
            return True
    return False


def _strip_leading_comments(sql: str) -> str:
    """Strip leading SQL comments (``--`` and ``/* */``) and whitespace.

    Also strips a leading UTF-8 BOM (``\\ufeff``), which ``str.strip()``
    does not treat as whitespace, for parity with ``sqlite3_prepare_v2``;
    otherwise a non-utf-8-sig-decoded SQL file desyncs the tx tracker.
    """
    s = sql.lstrip("﻿").strip()
    while True:
        if s.startswith("--"):
            newline = s.find("\n")
            if newline == -1:
                return ""
            s = s[newline + 1 :].strip()
        elif s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                # Unterminated block comment consumes everything.
                return ""
            s = s[end + 2 :].strip()
        else:
            break
    return s


def _parse_savepoint_name(after_keyword: str) -> str | None:
    """Extract a savepoint name from text following ``SAVEPOINT``.

    Handles only unquoted ASCII identifiers and lowercases the result
    (matching SQLite's bare-name fold). Returns ``None`` for quoted,
    backtick, bracketed, unicode, or leading-digit names, and for
    multi-statement input: tracking those would let a later RELEASE
    desync the local stack from the case-sensitive server. SA generates
    bare ``sa_savepoint_N`` names, so this covers the common path.
    """
    s = _strip_leading_comments(after_keyword)
    if not s or s[0] not in _BARE_IDENT_FIRST:
        return None
    end = 1
    while end < len(s) and s[end] in _BARE_IDENT_REST:
        end += 1
    # Reject trailing garbage after the identifier (SQLite parse-rejects it).
    if _strip_leading_comments(s[end:]):
        return None
    return s[:end].lower()


def _parse_release_name(after_keyword: str) -> str | None:
    """Extract a savepoint name following ``RELEASE`` / ``ROLLBACK TO``,
    stripping an optional leading ``SAVEPOINT`` keyword."""
    s = _strip_leading_comments(after_keyword)
    kw_len = len("SAVEPOINT")
    if s[:kw_len].upper() == "SAVEPOINT" and _is_keyword_boundary(s, kw_len):
        s = s[kw_len:]
    return _parse_savepoint_name(s)


def _is_no_tx_rollback_error(exc: BaseException) -> bool:
    """True if ``exc`` is the deterministic "no transaction is active"
    ROLLBACK reply: SQLITE_ERROR (primary 1) AND a wording fragment from
    :data:`dqlitewire.NO_TRANSACTION_MESSAGE_SUBSTRINGS`.

    Both conditions must hold so an unrelated error carrying the magic
    substring is not treated as benign.
    """
    if not isinstance(exc, OperationalError):
        return False
    code = getattr(exc, "code", None)
    if code is None or _primary_sqlite_code(code) != 1:  # SQLITE_ERROR
        return False
    # Match the un-truncated ``raw_message``, not ``str(exc)`` — a long
    # message could push the no-tx clause past the display truncation cap.
    raw = getattr(exc, "raw_message", None) or str(exc)
    msg = raw.lower()
    return any(s in msg for s in NO_TRANSACTION_MESSAGE_SUBSTRINGS)


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
from dqlitewire.messages.responses import _MAX_ADDRESS_SIZE as _WIRE_MAX_ADDRESS_SIZE  # noqa: E402

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
_parse_address = parse_address


def _connection_unclosed_warning(
    closed_flag: list[bool],
    connected_flag: list[bool],
    address: str,
    creator_pid: int,
    *,
    # Kwarg-default capture of module globals the body uses: ``Py_FinalizeEx``
    # phase 3 nulls them, so a finalize callback firing afterwards would raise
    # ``TypeError`` from the warnings machinery. ``get_current_pid`` is
    # deliberately NOT captured so fork-pid tests can patch it (the runtime
    # deref below is try-wrapped to absorb the shutdown-time TypeError).
    _warnings: Any = warnings,
    _contextlib: Any = contextlib,
    _sanitize_for_log: Any = sanitize_for_log,
) -> None:
    """Emit a ``ResourceWarning`` when a ``DqliteConnection`` is GC'd
    without ``await close()``.

    Three-flag gate: skip in a forked child (the snapshots belong to the
    parent), after ``close()`` (``closed_flag``), and for a
    never-connected instance (``connected_flag`` — would be a false positive).
    """
    if _warnings is None or _contextlib is None or _sanitize_for_log is None:
        # Phase-3 teardown nulled a captured global; bail.
        return
    # Read ``get_current_pid`` live so test patches are observed; the ``try``
    # absorbs the shutdown-time ``TypeError`` ('NoneType' is not callable).
    try:
        if get_current_pid() != creator_pid:
            return  # forked child — parent owns the lifecycle
    except Exception:
        return
    if closed_flag[0] or not connected_flag[0]:
        return
    with _contextlib.suppress(RuntimeError):
        # Sanitise before interpolation: a custom ``dial_func`` bypassing
        # ``parse_address`` could carry LF / U+2028 that splits a journald
        # record (CWE-117).
        _warnings.warn(
            f"DqliteConnection(address={_sanitize_for_log(str(address))!r}) "
            f"was garbage-collected without await close(). Call "
            f"``await conn.close()`` explicitly to release the "
            f"underlying socket promptly.",
            ResourceWarning,
            stacklevel=2,
        )


class DqliteConnection:
    """High-level async connection to a dqlite database.

    NOT thread-safe and not free-threaded-safe: use within a single
    asyncio loop, or ``asyncio.run_coroutine_threadsafe()`` to submit
    from other threads. Lazily binds to the first loop that uses it;
    cross-loop use raises ``InterfaceError``. Construct one per loop.
    """

    def __init__(
        self,
        address: str,
        *,
        database: str = "default",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        dial_timeout: float | None = None,
        attempt_timeout: float | None = None,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
        dial_func: DialFunc | None = None,
        max_message_size: int | None = None,
    ) -> None:
        """Initialize connection (does not connect yet).

        ``timeout`` is per-RPC-phase, applied to each phase independently, so one
        call can take up to N × ``timeout``; wrap in ``asyncio.timeout`` for a true
        deadline. ``dial_timeout`` (per-dial TCP establish) and ``attempt_timeout``
        (per-attempt dial + handshake + ``open_database``) each default to
        ``timeout``. ``max_total_rows`` and ``max_continuation_frames`` bound decode
        work; ``None`` disables. ``trust_server_heartbeat`` widens the per-read
        deadline to the server heartbeat (300 s hard cap). ``close_timeout`` bounds
        the best-effort transport drain in ``close()`` so an unresponsive peer
        cannot stall shutdown. ``dial_func`` replaces the default TCP path and its
        socket options (``None`` keeps the default). ``max_message_size`` is the
        inbound frame-size ceiling enforced by the wire ``ReadBuffer`` (default
        64 MiB).
        """
        validate_timeout(timeout)
        validate_timeout(
            close_timeout,
            name="close_timeout",
            min_value=CLOSE_TIMEOUT_FLOOR,
            min_value_rationale=CLOSE_TIMEOUT_FLOOR_RATIONALE,
        )
        if dial_timeout is not None:
            validate_timeout(dial_timeout, name="dial_timeout")
        if attempt_timeout is not None:
            validate_timeout(attempt_timeout, name="attempt_timeout")
        # Parse at construction so a bad address raises at config-load time,
        # not inside connect() where SA would mis-classify it as retryable.
        self._host, self._port = parse_address(address)
        self._address = address
        self._database = database
        self._timeout = timeout
        self._dial_timeout = dial_timeout if dial_timeout is not None else timeout
        self._attempt_timeout = attempt_timeout if attempt_timeout is not None else timeout
        self._close_timeout = close_timeout
        self._dial_func: DialFunc | None = dial_func
        self._max_total_rows = validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        # None falls back to the wire-layer default; the wire layer validates.
        self._max_message_size = max_message_size
        self._trust_server_heartbeat = trust_server_heartbeat
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False
        self._in_use = False
        # Task holding ``_in_use`` so ``_check_in_use`` can name claimant +
        # contender. Preserved across ``_invalidate`` like ``_in_use`` itself.
        self._in_use_claimant: asyncio.Task[Any] | None = None
        # "close() has been called" marker, distinct from ``is_connected``
        # ("transport alive"); set at the top of close() and never reset there.
        self._closed = False
        # Capture creator pid BEFORE the finalizer registration so its fork-pid
        # guard has a baseline. Fork-after-init is unsupported (shared socket,
        # parent-loop-bound asyncio primitives).
        self._creator_pid = os.getpid()
        # ResourceWarning finalizer for GC-without-close. List cells let the
        # closure read post-close values; ``creator_pid`` skips a forked child.
        self._closed_flag: list[bool] = [False]
        self._connected_flag: list[bool] = [False]
        self._finalizer: weakref.finalize[Any, Any] | None = weakref.finalize(
            self,
            _connection_unclosed_warning,
            self._closed_flag,
            self._connected_flag,
            self._address,
            self._creator_pid,
        )
        # Weakref so a closed-but-not-close()d loop is not pinned in memory.
        self._bound_loop_ref: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self._tx_owner: asyncio.Task[Any] | None = None
        # SAVEPOINT / RELEASE tracking: when the stack drains and the first
        # SAVEPOINT was an auto-begin, ``_in_transaction`` flips back to False.
        self._savepoint_stack: list[str] = []
        self._savepoint_implicit_begin = False
        # Set when a SAVEPOINT name is parser-rejected (quoted/unicode/etc): the
        # local stack stays empty but the server auto-began a tx, so the
        # pool-reset predicate ORs this in to roll the slot back on return.
        self._has_untracked_savepoint = False
        self._pool_released = False
        # Cause from ``_invalidate``; only meaningful while ``_protocol is
        # None``. ``connect()`` clears it so later "Not connected" errors
        # don't chain to an unrelated failure.
        self._invalidation_cause: BaseException | None = None
        # Bounded ``wait_closed`` drain scheduled by ``_invalidate`` so a
        # later ``close()`` can await it.
        self._pending_drain: asyncio.Task[None] | None = None

    @property
    def address(self) -> str:
        return self._address

    @property
    def _safe_address(self) -> str:
        """Sanitised ``self._address`` for exception messages (preserves
        LF/Tab; use :attr:`_log_safe_address` for log records).

        The address may be a server-supplied ``LeaderResponse`` left
        verbatim past ``parse_address``; routing through the sanitiser is
        CWE-117 defence-in-depth should hostname validation ever loosen.
        """
        return _sanitize_display_text(self._address)

    @property
    def _log_safe_address(self) -> str:
        """Logger-safe ``self._address``: layers ``sanitize_for_log`` so a
        peer cannot split a journald record via ``\\n`` / ``\\t`` (CWE-117)."""
        return sanitize_for_log(self._address)

    @property
    def is_connected(self) -> bool:
        return self._protocol is not None

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has been called (distinct from
        ``is_connected`` — a never-connected instance has both False)."""
        return self._closed

    def __repr__(self) -> str:
        # Sanitise the address: ``repr`` escapes C0 controls but not U+2028 /
        # bidi / ZW, which journald treats as record separators when a
        # downstream ``logger.X("%r", conn)`` interpolates this. CWE-117.
        state = "connected" if self._protocol is not None else "disconnected"
        safe_addr = sanitize_for_log(str(self._address))
        return (
            f"<DqliteConnection address={safe_addr!r} "
            f"database={self._database!r} {state} at 0x{id(self):x}>"
        )

    def __reduce__(self) -> NoReturn:
        # Holds a live socket + loop-bound StreamReader/Writer that cannot be
        # pickled; raise a clear driver-level TypeError instead of leaking
        # pickle's object-graph-walk error.
        raise TypeError(
            f"cannot pickle {type(self).__name__!r} object — holds a "
            f"live socket and a wire protocol wrapping loop-bound "
            f"StreamReader / StreamWriter; reconstruct from "
            f"configuration in the target process instead."
        )

    @property
    def in_transaction(self) -> bool:
        """True if a transaction is active, including a parser-rejected
        SAVEPOINT's untracked autobegin (so a caller branching on this
        cannot leak the tx). Mirrors stdlib ``sqlite3.in_transaction``."""
        return self._in_transaction or self._has_untracked_savepoint

    def _clear_savepoint_state(self) -> None:
        """Reset the three savepoint-tracking fields to their no-savepoint state."""
        self._savepoint_stack.clear()
        self._savepoint_implicit_begin = False
        self._has_untracked_savepoint = False

    def _clear_tx_state(self) -> None:
        """Reset all transaction + savepoint bookkeeping to the idle state."""
        self._in_transaction = False
        self._tx_owner = None
        self._clear_savepoint_state()

    def _find_savepoint_index(self, name: str) -> int:
        """Return the rightmost index of ``name`` (SQLite's LIFO duplicate rule)."""
        return len(self._savepoint_stack) - 1 - self._savepoint_stack[::-1].index(name)

    async def connect(self) -> None:
        """Establish connection to the database."""
        self._check_in_use()
        if self._protocol is not None:
            return

        # Claim _in_use INSIDE the try so a KeyboardInterrupt at the
        # bytecode boundary cannot leave the flag stuck True, and a
        # concurrent connect/close hits _check_in_use. See close() symmetry.
        try:
            self._in_use = True
            self._in_use_claimant = asyncio.current_task()
            await self._connect_impl()
        except BaseException:
            self._in_use = False
            self._in_use_claimant = None
            if self._protocol is None:
                self._bound_loop_ref = None
            raise
        else:
            self._in_use = False
            self._in_use_claimant = None

    async def _connect_impl(self) -> None:
        # Clear stale ``_invalidation_cause`` so a failed reconnect doesn't
        # surface a prior session's cause; the failure paths below don't
        # touch the field.
        self._invalidation_cause = None
        # Retire any drain task a prior ``_invalidate`` scheduled before the
        # slot is reused. Bounded re-snapshot loop because a racing
        # ``_invalidate`` (via ``call_soon_threadsafe``) can publish a fresh
        # task during ``await pending``. See ``_close_impl`` for the
        # cap-exhausted arm.
        resnapshot_cap = _CLOSE_RESNAPSHOT_CAP
        for _attempt in range(resnapshot_cap):
            pending = self._pending_drain
            self._pending_drain = None
            if pending is None or pending.done():
                break
            # Snapshot ``cancelling()`` before ``pending.cancel()``: it is
            # cumulative, so only a delta marks a fresh outer cancel during
            # ``await pending`` (vs our own cancel, which must be consumed).
            # Do NOT ``uncancel()`` — that would swallow the parent's cancel.
            self_task = asyncio.current_task()
            cancelling_before = self_task.cancelling() if self_task is not None else 0
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                cancelling_after = self_task.cancelling() if self_task is not None else 0
                if cancelling_after > cancelling_before:
                    raise
            except Exception:
                # Swallow non-cancel drain noise so reconnect proceeds;
                # BaseException would consume KI / SystemExit / outer cancel.
                pass
        else:
            # Cap exhausted (pathological ``_invalidate`` feedback loop):
            # observe + cancel the stuck task and WARN. Mirrors ``_close_impl``.
            stuck = self._pending_drain
            if stuck is not None and not stuck.done():
                if hasattr(stuck, "add_done_callback"):
                    from dqliteclient.cluster import _observe_drain_exception

                    stuck.add_done_callback(_observe_drain_exception)
                stuck.cancel()
            self._pending_drain = None
            logger.warning(
                "DqliteConnection._connect_impl: _pending_drain still set after "
                "%d re-snapshot iterations; cancelling residual task to avoid "
                "'Task was destroyed but it is pending' at GC. This indicates "
                "a pathological _invalidate feedback loop on connection id=%s.",
                resnapshot_cap,
                id(self),
            )

        self._bound_loop_ref = weakref.ref(asyncio.get_running_loop())
        # ``attempt_timeout`` wraps dial + handshake + open_database;
        # ``dial_timeout`` is nested inside as the per-dial budget. Use
        # ``asyncio.timeout`` cancel-scope (not ``wait_for``) so an outer
        # cancel mid-dial does not discard the ``(reader, writer)`` and
        # orphan the writer. ``writer = None`` so the finally has a name.
        writer = None
        try:
            async with asyncio.timeout(self._attempt_timeout):
                try:
                    async with asyncio.timeout(self._dial_timeout):
                        reader, writer = await open_connection(
                            self._address, dial_func=self._dial_func
                        )
                except TimeoutError as e:
                    raise DqliteConnectionError(
                        f"Connection to {self._safe_address} timed out"
                    ) from e
                except OSError as e:
                    raise DqliteConnectionError(
                        f"Failed to connect to {self._safe_address}: {e}"
                    ) from e

                self._protocol = DqliteProtocol(
                    reader,
                    writer,
                    timeout=self._timeout,
                    max_total_rows=self._max_total_rows,
                    max_continuation_frames=self._max_continuation_frames,
                    trust_server_heartbeat=self._trust_server_heartbeat,
                    address=self._address,
                    max_message_size=self._max_message_size,
                )
                # Protocol now owns the writer; null the local so the outer
                # finally is a no-op. If ``DqliteProtocol(...)`` itself raises,
                # ``writer`` is still set and the finally drains it.
                writer = None

                try:
                    await self._protocol.handshake()
                    logger.debug(
                        "connect: handshake ok address=%s client_id=%d",
                        self._log_safe_address,
                        self._protocol._client_id,
                    )
                    self._db_id = await self._protocol.open_database(self._database)
                    # Arm the unclosed-warning finalizer's connected gate.
                    self._connected_flag[0] = True
                    logger.debug(
                        "connect: db opened address=%s db_id=%d database=%r",
                        self._log_safe_address,
                        self._db_id,
                        self._database,
                    )
                    # A successful reconnect supersedes a prior invalidation.
                    self._invalidation_cause = None
                    # Clear sticky lifecycle markers so ``closed`` and the
                    # ResourceWarning gate reflect the reconnected slot. Placed
                    # AFTER the protocol publish so a concurrent ``closed``
                    # reader never sees ``_closed=False`` with ``_protocol``
                    # still None.
                    self._closed = False
                    self._closed_flag[0] = False
                    # Re-arm the finalizer ``close()`` detached.
                    if self._finalizer is None:
                        self._finalizer = weakref.finalize(
                            self,
                            _connection_unclosed_warning,
                            self._closed_flag,
                            self._connected_flag,
                            self._address,
                            self._creator_pid,
                        )
                except OperationalError as e:
                    await self._abort_protocol()
                    _is_leader_flip = e.code in LEADER_ERROR_CODES or (
                        e.code == _SQLITE_NOTFOUND
                        and (getattr(e, "raw_message", None) or e.message or "")
                        .lower()
                        .startswith(LEADER_LOST_DB_LOOKUP_SUBSTRING)
                    )
                    if _is_leader_flip:
                        # Leader-change during OPEN is transport-class, not a
                        # SQL error. Thread ``code`` / ``raw_message`` so the
                        # dbapi / SA classifiers get the coded signal (not just
                        # a substring). Sanitise the display message (CWE-117);
                        # ``raw_message`` stays verbatim for substring matchers.
                        raise DqliteConnectionError(
                            f"Node {self._safe_address} is no longer leader: "
                            f"{_sanitize_display_text(e.message)}",
                            code=e.code,
                            raw_message=e.raw_message,
                        ) from e
                    raise
                except ProtocolError as e:
                    # Handshake wire-decode failures are transport-class
                    # transient (peer mid-restart, partial reply). Rewrap as
                    # DqliteConnectionError so the cluster.py retry tuple
                    # fires. Thread ``code`` / ``raw_message`` so the verbatim
                    # text survives; sanitise the display message (CWE-117).
                    await self._abort_protocol()
                    raise DqliteConnectionError(
                        f"{WIRE_DECODE_FAILED_PREFIX} during handshake to "
                        f"{self._safe_address}: {_sanitize_display_text(str(e))}",
                        code=getattr(e, "code", None),
                        raw_message=getattr(e, "raw_message", None) or str(e),
                    ) from e
                except BaseException:
                    await self._abort_protocol()
                    raise
        except TimeoutError as e:
            # Per-attempt envelope exhausted. Reuse the "timed out" wording
            # the inner dial path produces so callers see a consistent
            # message regardless of which envelope tripped.
            await self._abort_protocol()
            raise DqliteConnectionError(
                f"Connection attempt to {self._safe_address} timed out"
            ) from e
        finally:
            # ``writer is None`` on the success path (handed off) and the
            # dial-failure path. If set, the dial succeeded but handoff was
            # interrupted, so drain here to avoid a leaked socket + reader Task.
            if writer is not None:
                writer.close()
                # Shielded bounded drain: the reader Task must be awaited or
                # it surfaces as "Task was destroyed but it is pending" at GC.
                # ``asyncio.shield`` keeps the inner drain alive across an
                # outer cancel (which still reaches the parent). Narrow
                # ``(OSError, TimeoutError)`` suppress surfaces unexpected
                # raises. ``asyncio.timeout`` (not ``wait_for``) preserves a
                # future ``wait_closed`` return value on cancel.
                close_timeout = self._close_timeout

                async def _drain() -> None:
                    async with asyncio.timeout(close_timeout):
                        await writer.wait_closed()

                inner_drain: asyncio.Task[None] = asyncio.ensure_future(_drain())
                # Local import to avoid an import cycle (cluster imports us).
                from dqliteclient.cluster import _observe_drain_exception

                inner_drain.add_done_callback(_observe_drain_exception)
                with contextlib.suppress(OSError, TimeoutError):
                    await asyncio.shield(inner_drain)

    async def close(self) -> None:
        """Close the connection. Idempotent.

        Nulls ``_protocol`` before awaiting ``wait_closed`` so a
        concurrent second close cannot re-enter the socket path. The
        drain is bounded by ``close_timeout`` (best-effort, since the
        local socket is already gone) so an unresponsive peer cannot
        stall ``engine.dispose()`` / SIGTERM shutdown.
        """
        # Fork-after-init: the socket FD is shared with the parent, so
        # ``writer.close()`` would FIN a socket the parent still uses. Flip
        # local state without touching the wire, then bail. Runs BEFORE the
        # ``_pool_released`` short-circuit so a pool-released conn inherited
        # across a fork still drops its parent-loop ``_pending_drain`` Task
        # (else "Task was destroyed but it is pending" + retained FD at GC).
        if get_current_pid() != self._creator_pid:
            # Drop every reference that crosses the fork boundary so child GC
            # keeps neither parent-loop primitives nor the inherited FD alive.
            self._closed = True
            self._closed_flag[0] = True
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            self._protocol = None
            self._db_id = None
            self._pending_drain = None
            self._clear_tx_state()
            self._bound_loop_ref = None
            # Clear the in-use guard so a child-process caller is not locked
            # out; the post-shortcut finally is not reached on this path.
            self._in_use = False
            self._in_use_claimant = None
            return
        # Pool-released conns already closed under pool ownership; no-op.
        if self._pool_released:
            return
        self._closed = True
        # Detach the finalizer — orderly shutdown, so no false-positive warning.
        self._closed_flag[0] = True
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        # Run the in-use guard BEFORE the ``_protocol is None`` early-return so
        # a connect() racing close() surfaces as InterfaceError, not a silent
        # no-op that would leak the eventual socket.
        self._check_in_use()
        # Claim ``_in_use`` INSIDE the try so a KI at the bytecode boundary
        # can't leave the flag stuck True and wedge the connection.
        try:
            self._in_use = True
            self._in_use_claimant = asyncio.current_task()
            await self._close_impl()
        finally:
            self._in_use = False
            self._in_use_claimant = None

    async def _close_impl(self) -> None:
        # Await any bounded drain ``_invalidate`` scheduled so the reader Task
        # exits cleanly (else "Task was destroyed but it is pending" at exit).
        # Bounded re-snapshot loop because a concurrent ``_invalidate`` (via
        # ``call_soon_threadsafe``) can create a fresh drain during
        # ``await pending``; cap at 3 to fail loudly on a feedback loop.
        resnapshot_cap = _CLOSE_RESNAPSHOT_CAP
        for _attempt in range(resnapshot_cap):
            pending = self._pending_drain
            self._pending_drain = None
            if pending is None or pending.done():
                break
            # We do NOT cancel ``pending`` here (unlike ``_connect_impl``);
            # our ``cancelling()`` counter only grows on a FRESH outer cancel,
            # which must propagate. A stable counter means a third party
            # cancelled it — consume and continue. Do NOT ``uncancel()``.
            self_task = asyncio.current_task()
            cancelling_before = self_task.cancelling() if self_task is not None else 0
            try:
                await pending
            except asyncio.CancelledError:
                cancelling_after = self_task.cancelling() if self_task is not None else 0
                if cancelling_after > cancelling_before:
                    raise  # fresh outer cancel — propagate
            except Exception:
                # Swallow transport drain noise; BaseException would consume KI.
                pass
        else:
            # Cap exhausted (pathological ``_invalidate`` feedback loop):
            # observe + cancel the stuck task and WARN, else it orphans at GC.
            stuck = self._pending_drain
            if stuck is not None and not stuck.done():
                # hasattr-guard tolerates adversarial test mocks.
                if hasattr(stuck, "add_done_callback"):
                    from dqliteclient.cluster import _observe_drain_exception

                    stuck.add_done_callback(_observe_drain_exception)
                stuck.cancel()
            self._pending_drain = None
            logger.warning(
                "DqliteConnection._close_impl: _pending_drain still set after "
                "%d re-snapshot iterations; cancelling residual task to avoid "
                "'Task was destroyed but it is pending' at GC. This indicates "
                "a pathological _invalidate feedback loop on connection id=%s.",
                resnapshot_cap,
                id(self),
            )
        # Clear tx bookkeeping (mirrors ``_invalidate``) before the
        # ``_protocol is None`` early-return so a raw-BEGIN-then-close leaves
        # no stale ``_in_transaction`` that would lie or trip the nested-tx
        # guard on reconnect. (Pool path clears via ``_reset_connection``.)
        self._clear_tx_state()
        # Drop the cached cause: its traceback can pin a large object graph
        # (e.g. a failed executemany's bind list) across close/reconnect.
        self._invalidation_cause = None
        # Clear the loop binding so a reconnect on a different loop is accepted.
        self._bound_loop_ref = None
        if self._protocol is None:
            return
        protocol = self._protocol
        self._protocol = None
        self._db_id = None
        protocol.close()
        # Bounded drain as a shielded Task so an outer cancel mid-``wait_closed``
        # does not discard the reader Task ("Task was destroyed..." at GC).
        # ``asyncio.timeout`` (not ``wait_for``) preserves a future return value.
        close_timeout = self._close_timeout

        async def _drain() -> None:
            async with asyncio.timeout(close_timeout):
                await protocol.wait_closed()

        inner_drain: asyncio.Task[None] = asyncio.ensure_future(_drain())
        # Local import to avoid an import cycle (cluster imports us).
        from dqliteclient.cluster import _observe_drain_exception

        inner_drain.add_done_callback(_observe_drain_exception)
        # Narrow suppression: a bounded drain may raise TimeoutError (slow
        # peer) / OSError (closed writer); CancelledError must propagate.
        try:
            await asyncio.shield(inner_drain)
        except OSError:
            # OSError subsumes TimeoutError.
            pass
        except Exception:
            logger.debug(
                "close: unexpected drain error for %s",
                self._log_safe_address,
                exc_info=True,
            )

    async def _abort_protocol(self) -> None:
        """Tear down a half-open protocol during a connect failure path.

        Close the writer, then drain ``wait_closed`` under the same bounded
        budget ``close()`` uses, shielded so an outer cancel does not orphan
        the reader Task ("Task was destroyed but it is pending" at GC).
        """
        protocol = self._protocol
        if protocol is None:
            return
        self._protocol = None
        protocol.close()
        # Shielded bounded drain; ``asyncio.timeout`` (not ``wait_for``)
        # preserves a future ``wait_closed`` return value on cancel.
        close_timeout = self._close_timeout

        async def _drain() -> None:
            async with asyncio.timeout(close_timeout):
                await protocol.wait_closed()

        inner_drain: asyncio.Task[None] = asyncio.ensure_future(_drain())
        # Local import to avoid an import cycle (cluster imports us).
        from dqliteclient.cluster import _observe_drain_exception

        inner_drain.add_done_callback(_observe_drain_exception)
        # Narrow suppression: TimeoutError (slow peer) / OSError (closed
        # writer) only; CancelledError must propagate.
        try:
            await asyncio.shield(inner_drain)
        except OSError:
            # OSError subsumes TimeoutError.
            pass
        except Exception:
            logger.debug(
                "_abort_protocol: unexpected drain error for %s",
                self._log_safe_address,
                exc_info=True,
            )

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Shield close() against an outer CancelledError so its bounded drain
        # is not cancelled mid-``wait_closed`` (orphaning the reader Task).
        # Distinguish a body-originated cancel (already ack'd by entering
        # __aexit__, suppressible) from a fresh outer cancel (must propagate)
        # via the ``cancelling()`` delta. Hoist into an explicit Task with an
        # observer so the shielded coroutine isn't orphaned on cancel.
        self_task = asyncio.current_task()
        cancelling_before = self_task.cancelling() if self_task is not None else 0
        from dqliteclient.cluster import _observe_drain_exception

        close_task = asyncio.ensure_future(self.close())
        close_task.add_done_callback(_observe_drain_exception)
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            cancelling_after = self_task.cancelling() if self_task is not None else 0
            if cancelling_after > cancelling_before:
                raise  # fresh outer cancel — propagate
            # Body-originated cancel: suppress; it reaches the caller via exc_val.
        except Exception as close_err:
            # A close-time exception must not supplant the body exception the
            # caller is already tracking (it would flip the primary class).
            # When the body exited cleanly, the close error is primary.
            if exc_val is not None:
                logger.debug(
                    "DqliteConnection.__aexit__: close raised after body exception",
                    exc_info=close_err,
                )
                return
            raise

    def _ensure_connected(self) -> tuple[DqliteProtocol, int]:
        if self._protocol is None or self._db_id is None:
            raise DqliteConnectionError("Not connected") from self._invalidation_cause
        return self._protocol, self._db_id

    def _check_in_use(self) -> None:
        """Raise on misuse, in this order: cross-process (fork) use,
        use after pool release, missing async context, wrong event
        loop, concurrent operation, or transaction owned by another
        task."""
        if get_current_pid() != self._creator_pid:
            raise InterfaceError(
                f"Connection used after fork; reconstruct from configuration "
                f"in the target process. (created in pid {self._creator_pid}, "
                f"current pid {get_current_pid()})"
            )
        if self._pool_released:
            raise InterfaceError(
                "This connection has been returned to the pool and can no longer "
                "be used directly. Acquire a new connection from the pool."
            )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError as e:
            # Preserve the RuntimeError on ``__cause__`` so a future second
            # ``get_running_loop`` shape ("loop is closing") stays visible.
            raise InterfaceError(
                "DqliteConnection must be used from within an async context."
            ) from e
        bound_loop = self._bound_loop_ref() if self._bound_loop_ref is not None else None
        if self._bound_loop_ref is None:
            # Lazy first-use bind so the guard covers connect()-skipping paths.
            self._bound_loop_ref = weakref.ref(current_loop)
        elif bound_loop is None:
            # Bound loop GC'd: refuse rather than silently rebind, which would
            # mask cross-loop misuse after the original loop was disposed.
            raise InterfaceError(
                "DqliteConnection is bound to a closed event loop. "
                "Reconstruct the connection in the new loop."
            )
        elif current_loop is not bound_loop:
            raise InterfaceError(
                "DqliteConnection is bound to a different event loop. "
                "Do not share connections across event loops or OS threads."
            )
        if self._in_use:
            # Report both claimant and contender for contention triage;
            # getattr-safe for ``__new__``-built test fixtures.
            claimant_repr = repr(getattr(self, "_in_use_claimant", None))
            current_repr = repr(asyncio.current_task())
            raise InterfaceError(
                "Cannot perform operation: another operation is in progress on this "
                f"connection (claimant: {claimant_repr}, current task: {current_repr}). "
                "DqliteConnection does not support concurrent coroutine access. "
                "Use a ConnectionPool to manage multiple concurrent operations."
            )
        if self._in_transaction and self._tx_owner is not None:
            current = asyncio.current_task()
            if current is not self._tx_owner:
                owner_repr = repr(self._tx_owner)
                current_repr = repr(current)
                raise InterfaceError(
                    "Cannot perform operation: connection is in a transaction owned "
                    f"by another task (owner: {owner_repr}, current: {current_repr}). "
                    "Each task should use its own connection from "
                    "the pool. Note: wrapping a connection call in "
                    "``asyncio.shield(conn.execute(...))`` creates a new task and "
                    "trips this check — shield the entire "
                    "``async with conn.transaction():`` block instead."
                )

    def _invalidate(self, cause: BaseException | None = None) -> None:
        """Mark the connection broken after an unrecoverable error.

        ``cause`` is remembered so a later "Not connected" can chain it.

        Does NOT clear ``_in_use``: that flag is owned by the task that
        claimed it. Clearing here (this can run out-of-band via
        ``call_soon_threadsafe``) would let a second task into a critical
        section while the original is mid-await; nulling ``_protocol``
        below unblocks it and its own ``finally`` clears the flag.

        Schedules a bounded ``wait_closed`` drain (close() is synchronous
        but wait_closed is a coroutine) so the reader Task doesn't outlive
        the connection as "Task was destroyed but it is pending".
        """
        if self._protocol is not None:
            proto = self._protocol
            # Connection may already be broken; suppress close errors
            with contextlib.suppress(Exception):
                proto.close()
            # Schedule a bounded drain only when a loop is running; some
            # callers (tests, inline error paths) run outside a loop.
            try:
                asyncio.get_running_loop()  # probe; ensure_future picks it up
            except RuntimeError:
                pass
            else:
                # ``asyncio.timeout`` (not ``wait_for``) preserves a future
                # ``wait_closed`` return value on cancel.
                async def _bounded_drain() -> None:
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(self._close_timeout):
                            await proto.wait_closed()

                # Cancel-and-detach any prior pending-drain before
                # overwriting, else the orphan trips "Task was destroyed
                # but it is pending" at GC.
                prior = self._pending_drain
                if prior is not None and not prior.done():
                    # Observe BEFORE cancel so a non-cancel escape doesn't
                    # surface "Task exception was never retrieved".
                    from dqliteclient.cluster import _observe_drain_exception

                    prior.add_done_callback(_observe_drain_exception)
                    prior.cancel()
                # Strong-ref so the task survives until close() awaits it.
                try:
                    new_task = asyncio.ensure_future(_bounded_drain())
                    # Observe the fresh task too, in case close() is never
                    # called (force_close_transport / dropped reference).
                    from dqliteclient.cluster import _observe_drain_exception

                    new_task.add_done_callback(_observe_drain_exception)
                    self._pending_drain = new_task
                except RuntimeError as schedule_err:
                    # ``ensure_future`` raises "Event loop is closed" during
                    # shutdown / dispose races. Bind to ``schedule_err`` (NOT
                    # ``cause``) so the ``as`` doesn't delete the ``cause``
                    # parameter used below. Log and move on with no drain.
                    logger.debug(
                        "Connection._invalidate: asyncio.ensure_future raised %s "
                        "while scheduling _bounded_drain; original cause preserved",
                        type(schedule_err).__name__,
                        exc_info=True,
                    )
                    self._pending_drain = None
        self._protocol = None
        self._db_id = None
        # ``_in_use`` is intentionally NOT cleared (see docstring): only the
        # claiming task may clear it. Clear the tx flags, though — an external
        # invalidation skips ``transaction()``'s finally, leaving a stale
        # ``_in_transaction`` / dead ``_tx_owner`` that the next caller reads
        # as a misleading "owned by another task".
        self._clear_tx_state()
        # Clear the loop binding so a fresh-loop reconnect is not rejected.
        self._bound_loop_ref = None
        # Preserve the FIRST cause: ``_run_protocol`` re-invalidates with the
        # synthetic "Not connected" wrapper, which would otherwise self-chain
        # and bury the real root cause.
        if cause is not None and self._invalidation_cause is None:
            self._invalidation_cause = cause
        # Mark "no leak" and detach the finalizer — the transport is gone, so
        # the GC ResourceWarning would be a false positive. ``_closed`` stays
        # unset so a follow-up ``close()`` is still idempotent; getattr
        # tolerates ``__new__`` test fixtures.
        closed_flag = getattr(self, "_closed_flag", None)
        if closed_flag is not None:
            closed_flag[0] = True
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
            self._finalizer = None

    @staticmethod
    def _validate_params(params: object) -> None:
        """Reject param containers that would scramble qmark positional
        binds: str/bytes/bytearray/memoryview (bind per char/byte),
        Mapping (named, not positional), set/frozenset (unordered)."""
        if params is None:
            return
        # ``DataError`` (a DqliteError) keeps the "every error is a
        # DqliteError" contract; a bare TypeError would leak past it.
        if isinstance(params, (str, bytes, bytearray, memoryview)):
            raise DataError(
                f"params must be a list or tuple, not {type(params).__name__!r}; "
                f"did you mean [value]?"
            )
        if isinstance(params, Mapping):
            raise DataError(
                "qmark paramstyle requires a sequence; got a mapping. "
                "Use a list or tuple positionally matching the ? placeholders."
            )
        if isinstance(params, (set, frozenset)):
            raise DataError(
                "qmark paramstyle requires an ordered sequence; got a set — "
                "iteration order is non-deterministic across runs."
            )
        # Reject single-shot iterators: the wire encoder calls ``len()``, so a
        # generator would raise deep inside encoding instead of here.
        if not isinstance(params, (list, tuple)) and not (
            hasattr(params, "__len__") and hasattr(params, "__getitem__")
        ):
            raise DataError(
                f"params must be a list or tuple, got {type(params).__name__!r}; "
                f"single-shot iterators are not supported (materialise via list(...))"
            )

    async def _run_protocol[T](self, fn: Callable[[DqliteProtocol, int], Awaitable[T]]) -> T:
        """Run a protocol operation with connection guards and standard
        error handling: invalidates on fatal errors, clears _in_use always."""
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        try:
            # Set ``_in_use`` INSIDE the try so a KI at the bytecode boundary
            # can't escape without running the finally that clears it — a
            # stuck True would permanently wedge the connection.
            self._in_use = True
            self._in_use_claimant = asyncio.current_task()
            return await fn(protocol, db_id)
        except _WireEncodeError as e:
            # Client-side encode error: bytes never reached the wire, so the
            # connection stays usable — do NOT invalidate. (A future streaming
            # encoder that writes mid-encode would break this and need
            # ``_invalidate(e)``.) The "wire encode failed: " prefix matches
            # the dbapi cursor's sibling arm for forensic distinction.
            raise DataError(f"wire encode failed: {e}") from e
        except (DqliteConnectionError, ProtocolError) as e:
            self._invalidate(e)
            raise
        except OperationalError as e:
            if e.code in LEADER_ERROR_CODES:
                self._invalidate(e)
            elif e.code == _SQLITE_NOTFOUND and (
                # Lower-case the haystack so an upstream capitalisation change
                # doesn't break the match.
                (getattr(e, "raw_message", None) or e.message or "")
                .lower()
                .startswith(LEADER_LOST_DB_LOOKUP_SUBSTRING)
            ):
                # Go-parity: SQLITE_NOTFOUND maps to ErrBadConn after leadership
                # loss. Substring-gated so the orthogonal LOOKUP_STMT NOTFOUND
                # (a stmt-id bug, not a transport flip) doesn't trigger it.
                self._invalidate(e)
            elif _primary_sqlite_code(e.code) in _TX_AUTO_ROLLBACK_PRIMARY_CODES:
                # Engine auto-rolled-back the tx; connection stays healthy.
                # Clear the local tx flags + savepoint stack so they don't lie.
                self._clear_tx_state()
            elif _primary_sqlite_code(e.code) == _SQLITE_BUSY and any(
                frag in (getattr(e, "raw_message", None) or e.message or "").lower()
                for frag in _RAFT_BUSY_MESSAGE_FRAGMENTS
            ):
                # SQLITE_BUSY has two origins sharing one code; only the
                # message distinguishes them. "checkpoint in progress" is the
                # one Raft-side case where the tx-state-clear is known safe
                # (other Raft-BUSY paths are indistinguishable — user retries).
                self._clear_tx_state()
                # Rewrap so SA's ``is_disconnect`` (which can't catch a coded
                # BUSY) recycles the pool slot via the connection-class arm.
                raise DqliteConnectionError(
                    f"raft-checkpoint reset the in-flight transaction: {e}",
                    code=e.code,
                    raw_message=getattr(e, "raw_message", None),
                ) from e
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit) as e:
            # Interrupted mid-op: we can't know how much round-trip completed,
            # so the wire state is unsafe to reuse — invalidate (writer close
            # FINs; the gateway tears down ``g->req`` on its next write). We
            # do NOT send INTERRUPT on a fresh socket: it's keyed on the same
            # connection's ``g->req``, so it would no-op after wasted dial.
            self._invalidate(e)
            raise
        except BaseExceptionGroup as eg:
            # A TaskGroup wraps sibling cancels in a group (PEP 654), which the
            # bare-class arm above does NOT match. Treat a group containing any
            # cancel-class child like that arm — invalidate. ``split`` recurses
            # so a buried/nested cancel is found. Non-cancel groups propagate
            # to the inner arms which encode the per-class policy.
            cancel_classes = (asyncio.CancelledError, KeyboardInterrupt, SystemExit)
            match, _rest = eg.split(cancel_classes)
            if match is not None:
                self._invalidate(eg)
            raise
        finally:
            self._in_use = False
            self._in_use_claimant = None

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement; return (last_insert_id, rows_affected).

        Sniffs the SQL prefix after a successful exec so ``in_transaction``
        stays in sync with raw ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``.
        """
        self._validate_params(params)
        try:
            result = await self._run_protocol(lambda p, db: p.exec_sql(db, sql, params))
        except OperationalError as e:
            # Split once so the deferred-FK check and conservative-flag set
            # can both consult the trailing piece of multi-statement input.
            pieces: list[str] | None = None
            trigger_stmt = sql
            is_multi_with_tx_verb = False
            if ";" in sql:
                pieces = _split_top_level_statements(sql)
                if pieces:
                    trigger_stmt = pieces[-1]
                if len(pieces) > 1 and any(_starts_with_tx_verb(p) for p in pieces):
                    is_multi_with_tx_verb = True

            # Deferred-FK auto-rollback: a code-19 on the OUTERMOST RELEASE or
            # a plain COMMIT/END rolls back the whole tx (per SQLite spec).
            # SQLITE_CONSTRAINT (19) is NOT in _TX_AUTO_ROLLBACK_PRIMARY_CODES
            # because a plain-INSERT CHECK violation does not auto-rollback,
            # so verb-condition the clear on the trailing piece.
            deferred_fk_cleared = False
            if (
                _primary_sqlite_code(e.code) == 19  # SQLITE_CONSTRAINT
                and self._sql_is_outermost_release_or_commit(trigger_stmt)
            ):
                self._clear_tx_state()
                deferred_fk_cleared = True

            # Multi-statement partial failure: an early piece may have opened
            # server-side tx state before the failing one, but the success-only
            # flag update below is skipped on raise. Conservatively flag
            # untracked state, unless an auto-rollback / deferred-FK clear
            # already ran (which would make this a redundant pool-reset).
            if (
                is_multi_with_tx_verb
                and not deferred_fk_cleared
                and _primary_sqlite_code(e.code) not in _TX_AUTO_ROLLBACK_PRIMARY_CODES
            ):
                self._has_untracked_savepoint = True
            raise
        self._update_tx_flags_from_sql(sql)
        return result

    def _sql_is_outermost_release_or_commit(self, sql: str) -> bool:
        """True if ``sql`` is a single ``COMMIT`` / ``END`` or a
        ``RELEASE`` of the OUTERMOST savepoint frame — the deferred-FK
        auto-rollback triggers (which tear down the whole tx)."""
        head = _strip_leading_comments(sql)
        if not head:
            return False
        upper = head.upper()
        # Gate COMMIT/END on ``_in_transaction``: a code-19 to a COMMIT outside
        # a tx is implausible, so don't zero state without the precondition.
        if upper.startswith("COMMIT") and _is_keyword_boundary(upper, len("COMMIT")):
            return self._in_transaction
        if upper.startswith("END") and _is_keyword_boundary(upper, len("END")):
            return self._in_transaction
        if upper.startswith("RELEASE") and _is_keyword_boundary(upper, len("RELEASE")):
            name = _parse_release_name(head[len("RELEASE") :])
            if name is not None and self._savepoint_stack and name == self._savepoint_stack[0]:
                return True
        return False

    def _update_tx_flags_from_sql(self, sql: str) -> None:
        """Update tx flags / savepoint stack after a successful execute, by
        prefix-sniffing the leading verb (a cheap heuristic, not a parser,
        like stdlib ``sqlite3``'s autocommit logic).

        Tracks SAVEPOINT / RELEASE because a bare SAVEPOINT outside a tx
        implicit-begins (matching RELEASE ends it); SAVEPOINTs nested in an
        explicit BEGIN keep ``_in_transaction`` True until COMMIT/ROLLBACK.
        ``ROLLBACK TO name`` pops frames above ``name`` but leaves the tx open.
        Multi-statement input is split and recursed so each piece is classified.
        """
        if ";" in sql:
            pieces = _split_top_level_statements(sql)
            if len(pieces) > 1:
                for piece in pieces:
                    self._update_tx_flags_from_sql(piece)
                return
            # len <= 1: empty, or a single piece with its tail ``;`` stripped.
            sql = pieces[0] if pieces else ""
        # Strip leading comments so the sniff sees past ``/* x */ BEGIN`` etc.
        head = _strip_leading_comments(sql)
        if not head:
            return
        upper = head.upper()
        if upper.startswith("BEGIN") and _is_keyword_boundary(upper, len("BEGIN")):
            if not self._in_transaction:
                self._in_transaction = True
                # Leave ``_tx_owner`` None for a raw BEGIN: the dbapi sync
                # facade runs each call as a fresh task, so binding the owner
                # would reject the next call as "owned by another task". Raw
                # BEGIN trusts the caller to serialise (stdlib sqlite3 parity).
            return
        if (
            upper.startswith("SAVEPOINT")
            and len(upper) > len("SAVEPOINT")
            and _is_keyword_boundary(upper, len("SAVEPOINT"))
        ):
            name = _parse_savepoint_name(head[len("SAVEPOINT") :])
            if name is not None:
                self._savepoint_stack.append(name)
                if not self._in_transaction and not self._has_untracked_savepoint:
                    # Implicit-begin: this savepoint is the outer frame
                    # (in_transaction=True, _tx_owner None like bare BEGIN).
                    # Skipped when an outer untracked SAVEPOINT already begat
                    # the tx — claiming it here would let this frame's RELEASE
                    # falsely flip _in_transaction while the server holds it.
                    self._in_transaction = True
                    self._savepoint_implicit_begin = True
            else:
                # Parser-rejected name: the server still creates the savepoint
                # (and may auto-begin), so flag untracked state for pool reset.
                self._has_untracked_savepoint = True
            return
        if (
            upper.startswith("RELEASE")
            and len(upper) > len("RELEASE")
            and _is_keyword_boundary(upper, len("RELEASE"))
        ):
            name = _parse_release_name(head[len("RELEASE") :])
            if name is None:
                # Parser-rejected name: we can't map it to a tracked frame, so
                # conservatively clear the stack (avoid ghost frames) and lock
                # ``_has_untracked_savepoint`` so pool reset keeps firing.
                self._savepoint_stack.clear()
                self._has_untracked_savepoint = True
                return
            if name in self._savepoint_stack:
                # RELEASE pops the named frame and all above it. Reverse-search
                # so SQLite's LIFO duplicate-name rule holds.
                idx = self._find_savepoint_index(name)
                del self._savepoint_stack[idx:]
                # Empty stack + autobegin means the implicit tx ends.
                if not self._savepoint_stack and self._savepoint_implicit_begin:
                    self._in_transaction = False
                    self._tx_owner = None
                    self._savepoint_implicit_begin = False
            else:
                # Valid name absent from the local stack — the server created
                # it via a path we didn't observe. Mirror the rejected branch:
                # clear the stack and lock the untracked flag.
                self._savepoint_stack.clear()
                self._has_untracked_savepoint = True
            return
        if upper.startswith("ROLLBACK"):
            # Bare ``lstrip()`` consumes all whitespace SQLite accepts (tab /
            # CR / FF), so ``ROLLBACK\tTO sp`` isn't misread as full ROLLBACK.
            after_upper = upper[len("ROLLBACK") :].lstrip()
            after_orig = head[len("ROLLBACK") :].lstrip()
            # Strip optional ``TRANSACTION`` before testing for ``TO`` so
            # ``ROLLBACK TRANSACTION TO SAVEPOINT sp`` is a savepoint rollback.
            tx_kw_len = len("TRANSACTION")
            if after_upper[:tx_kw_len] == "TRANSACTION" and _is_keyword_boundary(
                after_upper, tx_kw_len
            ):
                after_upper = after_upper[tx_kw_len:].lstrip()
                after_orig = after_orig[tx_kw_len:].lstrip()
            # ``ROLLBACK TO`` unwinds frames above the named savepoint but
            # leaves it (and the outer tx) active.
            if after_upper.startswith("TO") and _is_keyword_boundary(after_upper, 2):
                name = _parse_release_name(after_orig[len("TO") :])
                if name is None:
                    # Unlike RELEASE, ROLLBACK TO does not pop the named frame,
                    # only those above it. Leave the stack untouched (dropping
                    # frames could over-correct) and lock the untracked flag.
                    self._has_untracked_savepoint = True
                    return
                if name in self._savepoint_stack:
                    # Reverse-search for SQLite's LIFO duplicate-name rule.
                    idx = self._find_savepoint_index(name)
                    del self._savepoint_stack[idx + 1 :]
                else:
                    # Name absent locally: leave the stack untouched (frames
                    # above the unobserved target are unknown) and lock the flag.
                    self._has_untracked_savepoint = True
                return
            if self._in_transaction:
                self._clear_tx_state()
            else:
                # ROLLBACK with no active tx is a server no-op; defensively
                # clear all fields so a state-machine bug can't leak entries.
                self._clear_savepoint_state()
            return
        if upper.startswith("COMMIT") or upper.startswith("END"):
            # COMMIT / END close the outer transaction.
            verb = "COMMIT" if upper.startswith("COMMIT") else "END"
            if _is_keyword_boundary(upper, len(verb)):
                if self._in_transaction:
                    self._in_transaction = False
                    self._tx_owner = None
                # Clear unconditionally: the autobegin-deferral path can leave
                # a non-empty stack with _in_transaction=False that a COMMIT of
                # the server's autobegun tx would otherwise leave as a ghost.
                self._clear_savepoint_state()
                return

    async def query_raw(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query and return raw ``(column_names, rows)``.

        See ``query_raw_typed`` when per-column wire ``ValueType`` codes
        are also needed.
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))

    async def query_raw_typed(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """Return ``(column_names, column_types, row_types, rows)``.

        ``column_types`` are first-frame per-column wire ``ValueType`` tags;
        ``row_types`` is one tag-list per row (SQLite is dynamically typed, so
        a column's type can vary by row). Use ``query_raw`` when not needed.
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.query_sql_typed(db, sql, params))

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        self._validate_params(params)
        columns, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        if len(rows) < _FETCH_DICT_YIELD_EVERY:
            return [dict(zip(columns, row, strict=True)) for row in rows]
        # Large result: cede the loop every stride. ``strict=True`` keeps the
        # arity-mismatch ValueError.
        result: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            result.append(dict(zip(columns, row, strict=True)))
            if (i + 1) % _FETCH_DICT_YIELD_EVERY == 0:
                await asyncio.sleep(0)
        return result

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        """Execute a query and return results as list of lists."""
        self._validate_params(params)
        _, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        return rows

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute a query and return the first row.

        dqlite returns all matching rows over the wire; add ``LIMIT 1``
        for large result sets.
        """
        results = await self.fetch(sql, params)
        return results[0] if results else None

    async def fetchval(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Execute a query and return the first column of the first row."""
        self._validate_params(params)
        _, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        if rows and rows[0]:
            return rows[0][0]
        return None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Transaction context manager: BEGIN on entry, COMMIT on clean
        exit, ROLLBACK if the body raises.

        Cancellation: during the body, ROLLBACK is attempted; during
        COMMIT or a cancelled/failed ROLLBACK, the connection is
        invalidated so the pool discards it (server-side state ambiguous).
        CancelledError always propagates.

        Shielding caveat: ``asyncio.shield(conn.execute(...))`` inside the
        body makes a new task that ``_check_in_use`` rejects as "owned by
        another task" — shield the entire ``async with`` block instead.
        """
        # Misuse guard first so a forked child sees "used after fork", not the
        # misleading "owned by another task" branch below.
        self._check_in_use()
        # An untracked SAVEPOINT auto-begins a server tx without flipping
        # ``_in_transaction``; surface the SAVEPOINT root cause rather than a
        # wire-level "transaction within a transaction" or a repr(None) owner.
        if self._has_untracked_savepoint and not self._in_transaction:
            raise InterfaceError(
                "Cannot start transaction: a SAVEPOINT outside an explicit "
                "BEGIN is currently open on this connection (the SQLite "
                "engine has auto-begun a transaction). Issue COMMIT / "
                "ROLLBACK or RELEASE the outer SAVEPOINT before entering "
                "transaction()."
            )
        # Only the SAME task re-entering is "nested"; a sibling task gets the
        # "owned by another task" diagnostic (remedy: a separate connection).
        if self._in_transaction:
            if self._tx_owner is asyncio.current_task():
                raise InterfaceError(
                    "Nested transactions are not supported; use SAVEPOINT directly"
                )
            owner_repr = repr(self._tx_owner)
            raise InterfaceError(
                "Cannot start transaction: connection is in a transaction owned "
                f"by another task ({owner_repr}). Each task should use its own "
                "connection from the pool."
            )

        # Set the flags before the BEGIN await so a task switch mid-BEGIN
        # surfaces as the specific "nested transactions" error. The except
        # clears on any failure.
        self._in_transaction = True
        self._tx_owner = asyncio.current_task()
        try:
            await self.execute(_TRANSACTION_BEGIN_SQL)
        except BaseException:
            self._tx_owner = None
            self._in_transaction = False
            raise

        commit_attempted = False
        try:
            yield
            commit_attempted = True
            await self.execute(_TRANSACTION_COMMIT_SQL)
        except BaseException as exc:
            if commit_attempted:
                # COMMIT failed. Deterministic-rollback codes
                # (_TX_AUTO_ROLLBACK_PRIMARY_CODES, or code-19 deferred-FK on
                # COMMIT) leave a known no-tx state already cleared by
                # ``execute``, so the connection is reusable. Invalidate only
                # when the server-side state is genuinely ambiguous.
                deterministic_rollback = (
                    isinstance(exc, OperationalError)
                    and exc.code is not None
                    and (
                        _primary_sqlite_code(exc.code) in _TX_AUTO_ROLLBACK_PRIMARY_CODES
                        or _primary_sqlite_code(exc.code) == 19  # SQLITE_CONSTRAINT
                    )
                )
                if not deterministic_rollback:
                    # Pass the in-flight exception as cause so a later
                    # "Not connected" chains back to it.
                    self._invalidate(exc)
                    # Cancel-class exceptions MUST propagate verbatim
                    # (structured-concurrency). For other ambiguous shapes,
                    # raise AmbiguousCommitError (an OperationalError subclass)
                    # so retry middleware can treat the retry as at-least-once.
                    if not isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                        raw = getattr(exc, "raw_message", None) or str(exc)
                        code = getattr(exc, "code", None) or 0
                        raise AmbiguousCommitError(
                            f"COMMIT mid-flight failure; server-side commit state "
                            f"is unknown (may or may not have been applied and "
                            f"replicated): {exc}",
                            code,
                            raw_message=raw,
                        ) from exc
            else:
                # Body raised before COMMIT; try to roll back. If ROLLBACK
                # fails the tx state is unknowable, so invalidate (the pool
                # discards it). The body exception still propagates, except
                # cancellation which takes precedence.
                try:
                    await self.execute(_TRANSACTION_ROLLBACK_SQL)
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    # Rollback interrupted; tx state unknown — invalidate and
                    # propagate the higher-priority signal.
                    logger.debug(
                        "transaction(address=%s, id=%s): rollback was "
                        "cancelled mid-flight; connection invalidated; "
                        "propagating cancellation",
                        self._log_safe_address,
                        id(self),
                        exc_info=True,
                    )
                    self._invalidate()
                    raise
                except OperationalError as roll_exc:
                    # "No transaction is active" is the deterministic
                    # nothing-to-roll-back reply — the connection is healthy;
                    # preserve it and re-raise the body exception.
                    if _is_no_tx_rollback_error(roll_exc):
                        logger.debug(
                            "transaction(address=%s, id=%s): rollback "
                            "found no active transaction (server-side "
                            "tx already gone); preserving connection",
                            self._log_safe_address,
                            id(self),
                        )
                        # No active tx means the savepoint stack is gone too;
                        # clear it so pool reset doesn't re-issue a ROLLBACK.
                        self._clear_savepoint_state()
                    else:
                        logger.debug(
                            "transaction(address=%s, id=%s): rollback failed "
                            "with OperationalError; connection invalidated; "
                            "propagating original body exception",
                            self._log_safe_address,
                            id(self),
                            exc_info=True,
                        )
                        # Pass the rollback failure as the cause for a later
                        # "Not connected"; the body exception still propagates.
                        self._invalidate(roll_exc)
                except Exception as roll_exc:
                    # Non-OperationalError rollback failure: invalidate, then
                    # re-raise the original body exception below.
                    logger.debug(
                        "transaction(address=%s, id=%s): rollback failed; "
                        "connection invalidated; propagating original body "
                        "exception",
                        self._log_safe_address,
                        id(self),
                        exc_info=True,
                    )
                    self._invalidate(roll_exc)
            raise
        finally:
            # Defence-in-depth: the paths above already clear these; clearing
            # again keeps the invariant local to transaction()'s exit. Idempotent.
            self._clear_tx_state()
