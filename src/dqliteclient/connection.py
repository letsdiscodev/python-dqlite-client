"""High-level connection interface for dqlite."""

import asyncio
import contextlib
import ipaddress
import logging
import math
import re
import string
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Final

from dqliteclient.exceptions import (
    DataError,
    DqliteConnectionError,
    InterfaceError,
    OperationalError,
    ProtocolError,
)
from dqliteclient.protocol import (
    DqliteProtocol,
    _validate_positive_int_or_none,
)
from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES as _DEFAULT_MAX_CONTINUATION_FRAMES
from dqlitewire import DEFAULT_MAX_TOTAL_ROWS as _DEFAULT_MAX_TOTAL_ROWS
from dqlitewire import LEADER_ERROR_CODES, NO_TRANSACTION_MESSAGE_SUBSTRINGS
from dqlitewire import SQLITE_BUSY as _SQLITE_BUSY
from dqlitewire import TX_AUTO_ROLLBACK_PRIMARY_CODES as _TX_AUTO_ROLLBACK_PRIMARY_CODES
from dqlitewire import primary_sqlite_code as _primary_sqlite_code
from dqlitewire.exceptions import EncodeError as _WireEncodeError

__all__ = ["DqliteConnection"]

logger = logging.getLogger(__name__)

# Bare ``BEGIN`` opens an implicit ``DEFERRED`` transaction per SQLite's
# grammar default. dqlite's Raft FSM serializes transactions regardless
# of the qualifier, so DEFERRED / IMMEDIATE / EXCLUSIVE collapse to the
# same SERIALIZABLE semantics on dqlite — there is no benefit to
# emitting an explicit qualifier and doing so would diverge from the C
# and Go peer clients which also emit a bare ``BEGIN``. The constant
# pins the literal so a refactor can't silently upgrade to ``BEGIN
# IMMEDIATE`` without showing up in review.
_TRANSACTION_BEGIN_SQL: Final[str] = "BEGIN"

# ``COMMIT`` and ``ROLLBACK`` literals are pinned for the same reason as
# ``_TRANSACTION_BEGIN_SQL``: a refactor must not silently switch to a
# vendor-qualified form (``COMMIT TRANSACTION``) or a savepoint variant.
# Consistency between ``transaction()`` and the pool's reset-on-return
# path is a correctness invariant — if they diverge, a connection could
# be returned to the pool with a still-open server-side transaction.
_TRANSACTION_COMMIT_SQL: Final[str] = "COMMIT"
_TRANSACTION_ROLLBACK_SQL: Final[str] = "ROLLBACK"

# ``_TX_AUTO_ROLLBACK_PRIMARY_CODES`` is imported above from ``dqlitewire``
# (canonical home for SQLite-side constants). It contains the primary
# SQLite codes whose semantics imply the server-side transaction was
# automatically rolled back. Upstream ``leader.c`` polls
# ``sqlite3_txn_state`` after each exec and clears ``active_leader`` when
# the engine reports ``SQLITE_TXN_NONE``, so the cluster-side tx is gone
# for any of these primary codes:
#
#   * SQLITE_ABORT (4) — operation aborted, e.g. via sqlite3_interrupt.
#   * SQLITE_INTERRUPT (9) — query interrupted via INTERRUPT.
#   * SQLITE_IOERR (10) — and most extended IOERR variants (the leader-
#     change variants are caught earlier as ``LEADER_ERROR_CODES`` and
#     trigger a full ``_invalidate`` instead).
#   * SQLITE_CORRUPT (11).
#   * SQLITE_FULL (13).
#
# When ``_run_protocol`` sees one of these on a non-leader-class
# OperationalError, the underlying connection is still healthy but
# ``_in_transaction`` / ``_tx_owner`` must be cleared — otherwise the
# Python side reports True for ``in_transaction`` while the server
# reports tx-none, and the next user statement implicitly auto-begins
# a fresh transaction whose boundary the user code does not know about.


_BARE_IDENT_FIRST: Final[frozenset[str]] = frozenset(string.ascii_letters + "_")
_BARE_IDENT_REST: Final[frozenset[str]] = frozenset(string.ascii_letters + string.digits + "_")

# Coupled to dqlite-upstream/src/gateway.c failure() emissions for the
# Raft-side BUSY path that means "the in-flight write was not accepted;
# the server-side tx may already be gone." Currently the only such
# wording is "checkpoint in progress" (gateway.c emits it for
# checkpoint-contention BUSY); other Raft-BUSY paths are
# indistinguishable from engine-BUSY at the Python layer. Centralising
# the matcher here keeps the upstream-coupling explicit so a future
# rewording (or addition of a new Raft-BUSY message) is one-line update
# rather than a hunt through the classifier.
_RAFT_BUSY_MESSAGE_FRAGMENTS: Final[tuple[str, ...]] = ("checkpoint in progress",)


def _is_keyword_boundary(s: str, kw_len: int) -> bool:
    """True if position ``kw_len`` in ``s`` ends an SQL keyword.

    A keyword ends at end-of-string or at a character that is NOT a
    SQLite identifier-continuation character. ``str.isalnum`` is wrong
    here: SQLite's identifier tokenizer (``sqlite3IsIdChar``) treats
    ``_`` as an identifier character, but ``'_'.isalnum()`` is False —
    so an unfixed boundary check would split ``SAVEPOINT_foo`` into
    ``SAVEPOINT`` + ``_foo`` and silently push ``_foo`` onto the local
    savepoint stack while the server (correctly) creates ``SAVEPOINT_foo``.
    """
    return len(s) == kw_len or s[kw_len] not in _BARE_IDENT_REST


def _split_top_level_statements(sql: str) -> list[str]:
    """Split SQL on top-level ``;`` boundaries.

    The dqlite server's EXEC path supports multi-statement input — it
    iterates the statement list, executing each in turn (see
    ``dqlite-upstream/src/gateway.c`` ``handle_exec_sql_done_cb``). The
    transaction tracker's prefix-sniff sees only the leading verb, so
    ``execute("SAVEPOINT a; SAVEPOINT b;")`` would push only ``a``
    locally while the server pushes both. Splitting here lets the
    tracker re-classify each piece independently.

    SQLite tokenisation skipped while scanning for a top-level ``;``:

    * ``'...'`` single-quoted string literal (``''`` is an escaped ``'``)
    * ``"..."`` double-quoted identifier (``""`` is an escaped ``"``)
    * ``[...]`` square-bracket identifier (no escape — terminated by ``]``)
    * `````...````` backtick identifier (`````` is an escaped `````)
    * ``--`` to end-of-line — line comment
    * ``/* ... */`` — block comment

    Trigger-body block scope (``parse.y::trigger_cmd_list``): SQLite
    treats ``;`` inside a ``CREATE [TEMP|TEMPORARY] TRIGGER ... BEGIN
    ... END`` as inner-statement terminators that do NOT close the
    outer DDL. Track ``BEGIN..END`` depth ONLY after seeing the
    trigger preamble — a bare ``BEGIN`` is transaction-control and
    must still split.

    Returns whitespace-stripped non-empty pieces. Keep this small and
    self-contained — adding a real tokeniser is out of scope.
    """
    out: list[str] = []
    start = 0
    i = 0
    n = len(sql)
    # ``in_trigger_body``: inside a ``CREATE TRIGGER ... BEGIN`` body.
    # ``trigger_depth``: inner BEGIN..END nesting count (for compound
    # SQLite trigger bodies, e.g. ``BEGIN ... BEGIN ... END; END;``).
    in_trigger_body = False
    trigger_depth = 0
    # Position in the current statement piece where we last checked
    # for a CREATE TRIGGER preamble — only check ahead of, not behind,
    # to keep this O(n) per piece.
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
        # Track CREATE TRIGGER ... BEGIN entry and BEGIN..END nesting
        # so an inner ``;`` inside a trigger body does not close the
        # outer DDL. Only enter trigger-body mode after recognising
        # the full preamble; a bare ``BEGIN`` (transaction-control)
        # outside trigger mode must still permit ``;`` to split.
        if c.isalpha() and (i == 0 or not _is_word_char(sql[i - 1])):
            kw_end = i
            while kw_end < n and _is_word_char(sql[kw_end]):
                kw_end += 1
            kw = sql[i:kw_end].upper()
            if not in_trigger_body:
                # Look for ``CREATE TRIGGER ... BEGIN`` with optional
                # ``TEMP`` / ``TEMPORARY`` between CREATE and TRIGGER.
                if kw == "CREATE" and i >= trigger_scan_start:
                    j = _scan_for_trigger_begin(sql, kw_end, n)
                    if j > 0:
                        in_trigger_body = True
                        trigger_depth = 1
                        i = j
                        continue
            else:
                # Inside trigger body — track nested BEGIN..END so a
                # compound trigger (BEGIN ... BEGIN ... END; END;)
                # closes correctly.
                if kw == "BEGIN":
                    trigger_depth += 1
                    i = kw_end
                    continue
                if kw == "END":
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
        # When inside a trigger body, ``;`` is an inner-statement
        # terminator and does not split the outer DDL.
        i += 1
    tail = sql[start:].strip()
    if tail:
        out.append(tail)
    return out


def _is_word_char(c: str) -> bool:
    """True if ``c`` is part of a SQL keyword/identifier word."""
    return c.isalnum() or c == "_"


def _scan_for_trigger_begin(sql: str, after_create: int, n: int) -> int:
    """Look ahead from just after ``CREATE`` for the trigger preamble
    ``[TEMP|TEMPORARY] TRIGGER ... BEGIN`` and return the index just
    past the ``BEGIN`` keyword on success, or 0 (not a trigger).

    Skips over quoted identifiers, comments, and parenthesised
    sub-expressions (the ``WHEN (...)`` clause). Stops at any ``;``
    or end-of-input. Symmetric with the splitter's quote/comment
    handling so a ``CREATE TRIGGER`` inside a string literal is not
    matched (the outer splitter has already advanced past those
    constructs by the time the alpha-token branch fires).
    """
    i = after_create
    # Optional whitespace, then optional TEMP/TEMPORARY, then required
    # TRIGGER keyword.
    while i < n and sql[i].isspace():
        i += 1
    if i >= n:
        return 0
    j = i
    while j < n and _is_word_char(sql[j]):
        j += 1
    word = sql[i:j].upper()
    if word in ("TEMP", "TEMPORARY"):
        i = j
        while i < n and sql[i].isspace():
            i += 1
        j = i
        while j < n and _is_word_char(sql[j]):
            j += 1
        word = sql[i:j].upper()
    if word != "TRIGGER":
        return 0
    i = j
    # Now scan forward for the next standalone BEGIN at the same
    # nesting level, respecting quotes / comments / parens.
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
            # Reached the end of the would-be CREATE TRIGGER without a
            # BEGIN — not a trigger-body DDL after all (could be a
            # ``CREATE TRIGGER ... INSERT`` short-form trigger which
            # sqlite also supports without BEGIN..END). Bail out.
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


# Fixed-order tuple instead of a frozenset: iteration order is
# hash-seeded for frozensets, which would let a future verb that shares
# a prefix with an existing one (e.g. ``BEGINEXCLUSIVE``) produce
# order-dependent classification. Longest-first ordering keeps any
# future prefix-sharing safe — ``startswith`` matches the longest
# candidate first.
_TX_CONTROL_VERBS: tuple[str, ...] = (
    "SAVEPOINT",
    "ROLLBACK",
    "RELEASE",
    "COMMIT",
    "BEGIN",
    "END",
)


def _starts_with_tx_verb(stmt: str) -> bool:
    """True if ``stmt`` (already split off a multi-statement batch by
    ``_split_top_level_statements``) starts with a transaction-control
    verb. Used to distinguish "may have changed tx state" pieces from
    benign DML in the multi-statement-failure conservative-flag path.

    Strips leading comments + whitespace before the verb extraction;
    splits the verb on the keyword boundary helper to avoid mistaking
    ``BEGIN_foo`` (a bareword identifier) for the BEGIN keyword.
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

    SQLite accepts both comment styles as leading whitespace and runs
    the post-comment statement normally. The transaction tracker's
    prefix-sniff classifier needs to see past leading comments or it
    silently misses BEGIN / COMMIT / ROLLBACK / SAVEPOINT / RELEASE
    statements wrapped in annotations, leaving the local state out of
    step with the server. This helper is duplicated from the dbapi
    cursor's identical helper rather than introducing an inter-package
    import — the parser is small and stable.

    Also strips a leading UTF-8 BOM (``\\ufeff``) for parity with
    SQLite's ``sqlite3_prepare_v2``, which silently skips it. Python's
    ``str.strip()`` does NOT consider ``\\ufeff`` whitespace, so a SQL
    file imported via ``encoding='utf-8'`` (instead of ``utf-8-sig``)
    or written by PowerShell ``Set-Content`` / Notepad would otherwise
    desync the savepoint / transaction tracker.
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
                # Symmetric with the unterminated-``--`` branch: an
                # unterminated block comment consumes everything. SQLite
                # parse-rejects this shape; downstream callers all treat
                # an empty return as "no verb / no name", so collapsing
                # to "" matches the existing semantic surface.
                return ""
            s = s[end + 2 :].strip()
        else:
            break
    return s


def _parse_savepoint_name(after_keyword: str) -> str | None:
    """Extract a savepoint name from text following ``SAVEPOINT``.

    Handles unquoted ASCII SQLite identifiers (ASCII letters, digits,
    underscore; leading-non-digit) and lowercases the result so
    unquoted-identifier matching is case-insensitive, mirroring SQLite's
    identifier resolution for bare ASCII names.

    Returns ``None`` for shapes the prefix-sniff cannot reliably and
    safely parse:

    * Double-quoted identifiers — SQLite treats ``"..."`` names as
      case-sensitive, but the unquoted branch lowercases. Tracking
      a quoted form would let a later unquoted RELEASE collide
      against the lowercased entry and pop the local stack while
      the server (case-sensitive) refuses, leaving the tracker
      out of step with the server. Returning ``None`` here keeps
      the local stack untouched for the quoted frame; the autobegin
      flag also does not transition. Hand-written quoted SAVEPOINTs
      are exotic — SA generates ``sa_savepoint_N`` which is bare.
    * Backtick / square-bracket / unicode identifiers — Python's
      ``str.isalnum`` returns True for non-ASCII letters, but
      ``str.lower`` may normalise them differently than the server
      would. Reject up front so the local stack does not desync from
      the server's identifier-fold rules.
    * Leading-digit identifiers — SQLite parse-rejects them, so the
      tracker would push a name the server has not created.
    * Multi-statement input.
    """
    # Strip both leading whitespace AND embedded comments (SQLite
    # treats ``/* ... */`` and ``--`` as whitespace anywhere in the
    # token stream). Without the comment-strip, ``SAVEPOINT /* x */ sp``
    # falls into the "untracked" branch even though the server creates
    # the savepoint normally.
    s = _strip_leading_comments(after_keyword)
    if not s or s[0] not in _BARE_IDENT_FIRST:
        return None
    end = 1
    while end < len(s) and s[end] in _BARE_IDENT_REST:
        end += 1
    # Reject trailing garbage after the identifier. SQLite tolerates
    # whitespace and comments at this position but parse-rejects any
    # other trailing token; the success-only update at the call site
    # masks the runtime impact today, but lenient parsing here is a
    # forward-defence concern (a future caller using this helper
    # outside ``_update_tx_flags_from_sql`` would silently accept
    # ``foo extra junk`` as ``foo``).
    if _strip_leading_comments(s[end:]):
        return None
    return s[:end].lower()


def _parse_release_name(after_keyword: str) -> str | None:
    """Extract a savepoint name from text following ``RELEASE`` or
    ``ROLLBACK TO``.

    SQLite accepts both ``RELEASE name`` and ``RELEASE SAVEPOINT name``
    (likewise for ``ROLLBACK TO``). Strips an optional leading
    ``SAVEPOINT`` keyword before extracting the identifier.
    """
    # Strip leading whitespace AND embedded comments, mirroring
    # ``_parse_savepoint_name``. Handles ``RELEASE /* x */ SAVEPOINT
    # sp`` and ``RELEASE /* x */ sp`` shapes.
    s = _strip_leading_comments(after_keyword)
    kw_len = len("SAVEPOINT")
    if s[:kw_len].upper() == "SAVEPOINT" and _is_keyword_boundary(s, kw_len):
        s = s[kw_len:]
    return _parse_savepoint_name(s)


def _is_no_tx_rollback_error(exc: BaseException) -> bool:
    """True if ``exc`` is the deterministic "no transaction is active"
    reply from the server during a ROLLBACK.

    Recognised by SQLite primary code 1 (``SQLITE_ERROR``) plus the
    wording fragments listed in
    :data:`dqlitewire.NO_TRANSACTION_MESSAGE_SUBSTRINGS`. Both
    conditions must hold so a disk-full / constraint / IO error whose
    message happens to include the magic substring is not silently
    treated as benign.

    Used by the ``transaction()`` context manager and by the pool's
    ``_reset_connection`` to distinguish "server already auto-rolled
    back; preserve the slot" from a real ROLLBACK failure that
    requires invalidation. The substring list lives in the wire layer
    so this recogniser and the dbapi's ``_is_no_transaction_error``
    cannot drift apart on a server-side wording change.
    """
    if not isinstance(exc, OperationalError):
        return False
    code = getattr(exc, "code", None)
    if code is None or _primary_sqlite_code(code) != 1:  # SQLITE_ERROR
        return False
    msg = str(exc).lower()
    return any(s in msg for s in NO_TRANSACTION_MESSAGE_SUBSTRINGS)


# RFC 1035 hostname labels are ASCII letters, digits, and hyphen. We
# accept a dotted sequence of labels up to 253 chars total. Single
# labels (e.g. "localhost") are also accepted.
_HOSTNAME_LABEL_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)"
    r"(?:\.(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?))*$"
)


def _canonicalize_host(host: str, address: str) -> str:
    """Validate and canonicalize a host portion of an address.

    Accepts IPv4 literals, IPv6 literals (already unwrapped from
    brackets), and ASCII hostnames. Returns the canonical form:
    ``ipaddress.ip_address(h)`` for IP literals, lowercase for
    hostnames. Rejects credentials-like '@', whitespace/CRLF, and
    non-ASCII (IDN) hosts so a server-controlled redirect target
    cannot smuggle log-injection or DNS-rebinding vectors past the
    parser.
    """
    if not host:
        raise ValueError(f"Invalid address format: empty hostname in {address!r}")
    # Try IP literal first — IPv6 shorthand (``::1``) must canonicalize
    # so allowlists see one form regardless of how the peer wrote it.
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    # ASCII-only: reject IDN outright. dqlite's wire does not round-
    # trip punycode reliably, and non-ASCII hostnames are a common
    # homograph-attack vector.
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
    return host.lower()


def _validate_timeout(value: float, *, name: str = "timeout") -> float:
    """Validate a user-supplied timeout: positive, finite, not ``bool``.

    ``isinstance(True, int)`` is True and ``math.isfinite(True)`` is
    True, so a bare ``<= 0`` check lets ``timeout=True`` through as
    ``1.0``. Reject ``bool`` explicitly. Also reject ``inf`` / ``nan``
    so a misconfigured value fails at the caller's entry point rather
    than much later when it reaches ``asyncio.wait_for``.

    Shared across ``DqliteConnection``, ``ClusterClient``, and
    ``ConnectionPool`` so every entry point enforces the same contract.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r} (bool)")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return float(value)


def _parse_address(address: str) -> tuple[str, int]:
    """Parse a host:port address string, handling IPv6 brackets.

    Returns ``(canonical_host, port)``. IP literals are returned in
    ``ipaddress.ip_address``'s canonical form; hostnames are
    lowercased. Invalid hosts (credentials-like '@', whitespace/CRLF,
    non-ASCII, empty) raise ``ValueError``.
    """
    if address.startswith("["):
        # Bracketed IPv6: [host]:port
        if "]:" not in address:
            raise ValueError(
                f"Invalid IPv6 address format: expected '[host]:port', got {address!r}"
            )
        bracket_end = address.index("]")
        host = address[1:bracket_end]
        port_str = address[bracket_end + 2 :]  # Skip ']:
    else:
        if ":" not in address:
            raise ValueError(f"Invalid address format: expected 'host:port', got {address!r}")
        host, port_str = address.rsplit(":", 1)
        # Diagnose unbracketed IPv6 BEFORE attempting to parse the
        # port: ``"::1:abc"`` splits to host="::1", port_str="abc",
        # and the ``int()`` failure would surface as "invalid port"
        # when the real error is the missing brackets. Require the
        # host to contain ``:`` and NOT contain ``@`` — the latter
        # indicates a credentials-smuggle shape (rejected by
        # ``_canonicalize_host`` below with a more specific message).
        if ":" in host and "@" not in host:
            raise ValueError(
                f"IPv6 addresses must be bracketed: got {address!r}, expected '[host]:port'"
            )

    # Strict port parse: stdlib ``int()`` accepts whitespace, unary
    # ``+``, PEP 515 underscores, leading zeros, and Unicode digits.
    # An ``allowlist_policy`` comparing against a configured address
    # set would then fail to match when the peer redirects to a
    # non-canonical form (``"host: 9000 "``, ``"host:+9000"``,
    # ``"host:9_000"``) even though it's semantically the same port.
    # Restrict to plain ASCII digits (with an optional leading ``-``
    # so negative ports surface the "not in range" diagnostic rather
    # than a confusing "not a number" one).
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


class DqliteConnection:
    """High-level async connection to a dqlite database.

    Thread safety: this class is NOT thread-safe. All operations must be
    performed within a single asyncio event loop. Do not share instances
    across OS threads or event loops. To submit work from other threads,
    use ``asyncio.run_coroutine_threadsafe()`` — the coroutines execute
    safely in the event loop thread. Free-threaded Python (no-GIL) is
    not supported.
    """

    def __init__(
        self,
        address: str,
        *,
        database: str = "default",
        timeout: float = 10.0,
        max_total_rows: int | None = _DEFAULT_MAX_TOTAL_ROWS,
        max_continuation_frames: int | None = _DEFAULT_MAX_CONTINUATION_FRAMES,
        trust_server_heartbeat: bool = False,
        close_timeout: float = 0.5,
    ) -> None:
        """Initialize connection (does not connect yet).

        Args:
            address: Node address in "host:port" format
            database: Database name to open
            timeout: Per-RPC-phase timeout in seconds. The same budget
                is applied to each phase of an operation independently
                (send, then read, then any continuation drain), so a
                single high-level call (e.g. a ``query_sql`` returning
                a large continuation-paginated result) can take up to
                roughly N × ``timeout`` end-to-end. To enforce a true
                end-to-end deadline, wrap the call in
                ``asyncio.timeout(...)`` or ``asyncio.wait_for(...)``.
            max_total_rows: Cumulative row cap across continuation
                frames for a single query. Prevents a slow-drip server
                from keeping the client alive indefinitely within the
                per-operation deadline. Set to ``None`` to disable.
            max_continuation_frames: Maximum number of continuation
                frames in a single query result. Caps the per-query
                Python-side decode work a hostile server can inflict
                by sending many 1-row frames. Set to
                ``None`` to disable.
            trust_server_heartbeat: When True, widen the per-read
                deadline to the server-advertised heartbeat (subject
                to a 300 s hard cap). When False (default), ``timeout``
                is authoritative — the server value cannot amplify it.
            close_timeout: Budget in seconds for the transport-drain
                half of ``close()``. After ``writer.close()`` the
                local side of the socket is gone; ``wait_closed``
                is best-effort cleanup. An unresponsive peer must not
                stall ``engine.dispose()`` or SIGTERM shutdown, so
                the drain is bounded by this value.
        """
        _validate_timeout(timeout)
        _validate_timeout(close_timeout, name="close_timeout")
        # Parse at construction so a misconfigured address (typoed DSN,
        # invalid port, unbracketed IPv6) raises ValueError at the
        # operator's config-load site rather than inside connect(),
        # where SA's is_disconnect substring scan would mis-classify
        # it as a retryable transport failure and loop.
        self._host, self._port = _parse_address(address)
        self._address = address
        self._database = database
        self._timeout = timeout
        self._close_timeout = close_timeout
        self._max_total_rows = _validate_positive_int_or_none(max_total_rows, "max_total_rows")
        self._max_continuation_frames = _validate_positive_int_or_none(
            max_continuation_frames, "max_continuation_frames"
        )
        self._trust_server_heartbeat = trust_server_heartbeat
        self._protocol: DqliteProtocol | None = None
        self._db_id: int | None = None
        self._in_transaction = False
        self._in_use = False
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._tx_owner: asyncio.Task[Any] | None = None
        # Savepoint stack and implicit-begin flag for SAVEPOINT / RELEASE
        # tracking. ``_update_tx_flags_from_sql`` pushes a SAVEPOINT name
        # on entry, pops on RELEASE / ROLLBACK TO. When the stack drains
        # and the first SAVEPOINT was an auto-begin (SQLite implicitly
        # begins a transaction when SAVEPOINT runs outside an active
        # one), ``_in_transaction`` is flipped back to False.
        self._savepoint_stack: list[str] = []
        self._savepoint_implicit_begin = False
        # Set to True whenever ``_update_tx_flags_from_sql`` observes a
        # SAVEPOINT verb whose name the parser cannot represent
        # (``_parse_savepoint_name`` returns ``None`` — quoted identifier,
        # backtick / square-bracket, unicode, leading-digit). The local
        # stack stays empty by design (case-sensitivity trade-off, see
        # ``_parse_savepoint_name`` docstring), but the server has
        # auto-begun a transaction. The pool-reset predicate ORs this
        # flag in so the slot is rolled back on return — without it, a
        # bare ``SAVEPOINT "Foo"`` issued without a preceding BEGIN
        # would leak the autobegun transaction across pool acquirers.
        self._has_untracked_savepoint = False
        self._pool_released = False
        # Cause recorded by ``_invalidate(cause=...)``; only meaningful
        # while ``_protocol is None``. ``connect()`` clears it on a
        # successful re-handshake so later "Not connected" errors don't
        # chain to an ancient unrelated failure.
        self._invalidation_cause: BaseException | None = None
        # Tracks the bounded ``wait_closed`` drain scheduled by
        # ``_invalidate`` so a subsequent ``close()`` can await it and
        # keep the reader task from outliving the connection.
        self._pending_drain: asyncio.Task[None] | None = None

    @property
    def address(self) -> str:
        """Get the connection address."""
        return self._address

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._protocol is not None

    def __repr__(self) -> str:
        state = "connected" if self._protocol is not None else "disconnected"
        return f"<DqliteConnection address={self._address!r} database={self._database!r} {state}>"

    @property
    def in_transaction(self) -> bool:
        """Check if a transaction is active.

        Returns True when an explicit ``BEGIN`` / SAVEPOINT-autobegin
        is in flight OR when a parser-rejected SAVEPOINT (quoted,
        backtick, square-bracketed, unicode, leading-digit) auto-
        began a transaction the local stack tracker cannot model.
        Mirrors stdlib ``sqlite3.Connection.in_transaction`` semantics
        and aligns the property with the pool-reset predicate so
        callers branching on the property cannot leak the autobegun
        tx by skipping a ``commit()`` / ``rollback()``.
        """
        return self._in_transaction or self._has_untracked_savepoint

    async def connect(self) -> None:
        """Establish connection to the database."""
        self._check_in_use()
        if self._protocol is not None:
            return

        # Claim _in_use synchronously so a concurrent connect() / close()
        # on the same instance hits _check_in_use() and raises
        # InterfaceError("another operation in progress") instead of
        # racing through the pending-drain await below. Pool callers
        # are already serialized via _lock; direct DqliteConnection
        # users (calling await conn.connect() directly) need this guard
        # to avoid concurrent-connect races that orphan one of two
        # half-built protocols. See companion close() symmetry.
        self._in_use = True
        try:
            await self._connect_impl()
        except BaseException:
            # On failure, clear _in_use AND _bound_loop if no protocol
            # was published. Mirrors the original failure-path discipline.
            self._in_use = False
            if self._protocol is None:
                self._bound_loop = None
            raise
        else:
            self._in_use = False

    async def _connect_impl(self) -> None:
        # If a prior ``_invalidate`` scheduled a bounded drain task,
        # retire it here before the slot gets reused. Leaving the
        # previous task in place would let a second invalidate at
        # line ~483 overwrite the slot without cancelling or awaiting
        # it, breaking the "strong ref so close() can await it"
        # discipline documented on that assignment.
        pending = self._pending_drain
        if pending is not None:
            if not pending.done():
                pending.cancel()
                # Awaiting a cancelled task raises ``CancelledError``;
                # that is the cancel WE delivered and must be consumed
                # here so connect() can proceed. But an outer
                # ``task.cancel()`` may have also landed on the current
                # task — distinguish via ``Task.cancelling()`` (a
                # READ-only counter of cancels delivered to the current
                # task). Our own ``pending.cancel()`` cancelled the
                # *inner* drain task and propagated CancelledError up
                # through ``await pending``, but does NOT increment the
                # current task's own cancel count; an outer
                # ``task.cancel()`` against this coroutine, by contrast,
                # increments ``cancelling()`` to >= 1. If > 0, the
                # cancel was outer; let it propagate to the next
                # checkpoint cleanly. We deliberately do NOT call
                # ``Task.uncancel()`` here — that decrements the
                # counter and would consume the outer cancel, leaving
                # ``connect()`` to silently open a TCP connection the
                # parent intended to abort.
                try:
                    await pending
                except asyncio.CancelledError:
                    # The CancelledError came either from our own
                    # ``pending.cancel()`` (which we want to consume
                    # so connect() can proceed) OR from an outer
                    # ``task.cancel()`` that propagated through our
                    # fut_waiter. ``Task.cancelling()`` distinguishes:
                    # it counts cancels delivered TO the current task,
                    # which our own ``pending.cancel()`` does not
                    # increment. If > 0, the cancel was outer; let
                    # it propagate to the next checkpoint.
                    self_task = asyncio.current_task()
                    if self_task is not None and self_task.cancelling() > 0:
                        raise
                except Exception:
                    pass
            self._pending_drain = None

        self._bound_loop = asyncio.get_running_loop()
        # ``_in_use`` was claimed synchronously in ``connect()`` above
        # (before any await checkpoint) so concurrent connect/close
        # calls on the same instance are rejected by ``_check_in_use``.
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._timeout,
            )
        except TimeoutError as e:
            raise DqliteConnectionError(f"Connection to {self._address} timed out") from e
        except OSError as e:
            raise DqliteConnectionError(f"Failed to connect to {self._address}: {e}") from e

        try:
            self._protocol = DqliteProtocol(
                reader,
                writer,
                timeout=self._timeout,
                max_total_rows=self._max_total_rows,
                max_continuation_frames=self._max_continuation_frames,
                trust_server_heartbeat=self._trust_server_heartbeat,
                address=self._address,
            )
        except BaseException:
            # Protocol construction is currently limited to argument
            # validation, which ``DqliteConnection.__init__`` already
            # enforces — but if it ever raises (now or through future
            # refactors), ``_abort_protocol`` is a no-op until
            # ``self._protocol`` is assigned, so reader/writer would
            # be leaked. Close the transport defensively.
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=self._close_timeout)
            raise

        try:
            await self._protocol.handshake()
            logger.debug(
                "connect: handshake ok address=%s client_id=%d",
                self._address,
                self._protocol._client_id,
            )
            self._db_id = await self._protocol.open_database(self._database)
            logger.debug(
                "connect: db opened address=%s db_id=%d database=%r",
                self._address,
                self._db_id,
                self._database,
            )
            # Clear any stale cause recorded by a prior ``_invalidate``.
            # The field is only meaningful while ``_protocol is None``;
            # a successful reconnect supersedes it. Without this,
            # a later silent invalidation (``_invalidate()`` with no
            # cause) would produce "Not connected" errors whose
            # ``__cause__`` chain points back at an unrelated historical
            # failure — misleading operators reading logs.
            self._invalidation_cause = None
        except OperationalError as e:
            await self._abort_protocol()
            if e.code in LEADER_ERROR_CODES:
                # Leader-change errors during OPEN are transport-level
                # problems — the caller needs to reconnect elsewhere, not
                # treat this as a SQL error.
                raise DqliteConnectionError(
                    f"Node {self._address} is no longer leader: {e.message}"
                ) from e
            raise
        except BaseException:
            await self._abort_protocol()
            raise

    async def close(self) -> None:
        """Close the connection.

        Idempotent: safe to call on an already-closed or pool-released
        connection. Null ``_protocol`` before awaiting ``wait_closed``
        so a concurrent second close cannot re-enter the socket-close
        path.

        The transport drain (``wait_closed``) is bounded by
        ``close_timeout``. ``close()`` is the hot path on every pool
        release and on ``engine.dispose()`` / SIGTERM shutdown; an
        unresponsive peer must not be able to stall shutdown by
        refusing to acknowledge a FIN. The local side of the socket
        is already closed after ``writer.close()`` — the remaining
        wait is best-effort cleanup, not correctness-critical.
        """
        # Pool-released connections are never in_use for close(); their
        # close path has already run under pool ownership.
        if self._pool_released:
            return
        # Run the in-use guard BEFORE the ``_protocol is None``
        # early-return so a concurrent ``connect()`` racing with
        # ``close()`` surfaces as ``InterfaceError`` instead of a silent
        # no-op. Without this, close() returning while connect() is
        # suspended in ``asyncio.open_connection`` would leak the
        # eventual socket — connect() publishes _protocol only on
        # success, so at the race moment _protocol is None and
        # close() would silently return.
        self._check_in_use()
        # Claim ``_in_use`` synchronously (before any await) so a
        # concurrent ``connect()`` / ``close()`` on the same instance
        # is rejected by ``_check_in_use``. Symmetric with ``connect()``.
        self._in_use = True
        try:
            await self._close_impl()
        finally:
            self._in_use = False

    async def _close_impl(self) -> None:
        # ``_invalidate`` may have scheduled a bounded drain task on
        # the writer it just closed. Await it so the reader task exits
        # cleanly; otherwise Python logs "Task was destroyed but it is
        # pending" at interpreter shutdown. Drop the slot either way
        # so a subsequent close/connect cycle starts fresh — clearing
        # only inside the ``_protocol is None`` branch left a done-task
        # reference on the happy path indefinitely, inconsistent with
        # ``connect()``'s own ``_pending_drain = None`` symmetry.
        pending = self._pending_drain
        self._pending_drain = None
        if pending is not None and not pending.done():  # pragma: no cover
            # Defensive: the pending bounded-drain task is set only
            # by ``_invalidate`` and is normally already-done by
            # the time ``close()`` runs (the drain is bounded by
            # ``close_timeout`` and the close path is the second
            # caller). Reaching here requires the rare race where
            # ``_invalidate`` fires between the snapshot read and
            # the second caller's resumption — verified by code
            # review, not coverage.
            #
            # Suppress ``BaseException`` (not just ``Exception``) so a
            # ``CancelledError`` delivered during ``await pending``
            # does not propagate out of close() before the protocol
            # tear-down at lines below runs — leaking ``_protocol``.
            # Symmetric with ``connect()``'s pending-retire path which
            # already uses ``BaseException``. The user's outer cancel
            # gets re-delivered at the next await boundary inside
            # close (e.g., ``writer.wait_closed`` below).
            with contextlib.suppress(BaseException):
                await pending
        # Mirror ``_invalidate``'s atomic clear of the transaction
        # bookkeeping. Without this, a raw ``BEGIN`` followed by an
        # explicit ``close()`` and a reconnect on the same instance
        # leaves ``_in_transaction=True`` (and possibly a stale
        # ``_tx_owner``) — the next caller's ``in_transaction`` lies
        # about server-side state, and ``transaction()`` trips the
        # "Nested transactions are not supported" guard with no real
        # nesting happening. Place the clear before the
        # ``_protocol is None`` early-return so an already-closed
        # connection that still carries stale flags (e.g. from a prior
        # raw BEGIN whose close was concurrent with this one) gets
        # them scrubbed too. Pool-released connections take the
        # early-return at the top of the method, so the pool's
        # ``_reset_connection`` remains the canonical clear for that
        # path; this clear covers direct ``DqliteConnection`` users.
        self._in_transaction = False
        self._tx_owner = None
        self._savepoint_stack.clear()
        self._savepoint_implicit_begin = False
        self._has_untracked_savepoint = False
        # Clear the loop binding so a subsequent ``connect()`` on a
        # different event loop is accepted by ``_check_in_use``. The
        # failed-connect path already clears this in ``connect()``'s
        # finally block, but the successful-close path was missing the
        # symmetry. Mirrors the loop-reset done by the dbapi-async
        # adapter on close (``done/ISSUE-159``).
        self._bound_loop = None
        if self._protocol is None:
            return
        protocol = self._protocol
        self._protocol = None
        self._db_id = None
        protocol.close()
        # Narrow the suppression: a bounded wait on the transport
        # drain can legitimately raise TimeoutError (slow peer) or
        # OSError (already-closed writer). Anything else — especially
        # CancelledError from an outer ``asyncio.timeout`` scope — must
        # propagate so structured-concurrency cancellation semantics
        # remain intact. DEBUG-log unexpected Exceptions for
        # diagnostics; do not swallow.
        try:
            await asyncio.wait_for(protocol.wait_closed(), timeout=self._close_timeout)
        except OSError:
            # OSError subsumes TimeoutError, so the single OSError
            # entry covers the slow-peer / already-closed-writer
            # cases.
            pass
        except Exception:
            logger.debug(
                "close: unexpected drain error for %s",
                self._address,
                exc_info=True,
            )

    async def _abort_protocol(self) -> None:
        """Tear down a half-open protocol during a connect failure path.

        Close the writer, then give ``wait_closed`` the same bounded
        drain budget ``close()`` uses. Both sites share the same
        reasoning: the socket is already closed on our side, and the
        best-effort drain must not stall when the peer is
        unresponsive.
        """
        protocol = self._protocol
        if protocol is None:
            return
        self._protocol = None
        protocol.close()
        # Narrow the suppression: a bounded wait on the transport drain
        # can legitimately raise TimeoutError (slow peer) or OSError
        # (already-closed writer). Anything else — especially
        # CancelledError from an outer ``asyncio.timeout`` scope — must
        # propagate so structured-concurrency cancellation semantics
        # remain intact. DEBUG-log an unexpected Exception for
        # diagnostics; do not swallow.
        try:
            await asyncio.wait_for(protocol.wait_closed(), timeout=self._close_timeout)
        except OSError:
            # OSError subsumes TimeoutError, so the single OSError
            # entry covers the slow-peer / already-closed-writer
            # cases.
            pass
        except Exception:
            logger.debug(
                "_abort_protocol: unexpected drain error for %s",
                self._address,
                exc_info=True,
            )

    async def __aenter__(self) -> "DqliteConnection":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def _ensure_connected(self) -> tuple[DqliteProtocol, int]:
        """Ensure we're connected and return protocol and db_id."""
        if self._protocol is None or self._db_id is None:
            raise DqliteConnectionError("Not connected") from self._invalidation_cause
        return self._protocol, self._db_id

    def _check_in_use(self) -> None:
        """Raise on misuse: wrong event loop, concurrent access, or use after pool release."""
        if self._pool_released:
            raise InterfaceError(
                "This connection has been returned to the pool and can no longer "
                "be used directly. Acquire a new connection from the pool."
            )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            raise InterfaceError(
                "DqliteConnection must be used from within an async context."
            ) from None
        if self._bound_loop is None:
            # Lazily bind on first use so the guard is always active, even
            # for bare-instantiation / mocked-protocol patterns that skip
            # connect().
            self._bound_loop = current_loop
        elif current_loop is not self._bound_loop:
            raise InterfaceError(
                "DqliteConnection is bound to a different event loop. "
                "Do not share connections across event loops or OS threads."
            )
        if self._in_use:
            current_repr = repr(asyncio.current_task())
            raise InterfaceError(
                "Cannot perform operation: another operation is in progress on this "
                f"connection (current task: {current_repr}). DqliteConnection does "
                "not support concurrent coroutine access. Use a ConnectionPool to "
                "manage multiple concurrent operations."
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
        """Mark the connection as broken after an unrecoverable error.

        If ``cause`` is provided, it is remembered so a later caller that
        hits "Not connected" can chain it as ``__cause__`` for diagnostics.

        Also clears the ``_in_use`` slot: invalidation can be invoked
        out-of-band (e.g. scheduled from the dbapi sync-timeout path
        via ``call_soon_threadsafe``), bypassing ``_run_protocol``'s
        finally that normally resets the flag. An invalidated connection
        is no longer holding a meaningful in-flight operation, so the
        flag and the liveness state must stay consistent — otherwise
        the next call deterministically raises "another operation is
        in progress" on a connection that is in fact dead.

        Synchronous writer-close + async bounded drain: ``protocol.close()``
        is synchronous (writer.close()), but ``wait_closed()`` is a
        coroutine. Without a drain, a subsequent ``close()`` early-
        returns on ``_protocol is None`` and the reader task that
        ``asyncio.open_connection`` spawned stays pending until GC,
        producing the familiar ``"Task was destroyed but it is pending"``
        noise on interpreter exit. Schedule a bounded drain task and
        remember it on ``self`` so ``close()`` can await it.
        """
        if self._protocol is not None:
            proto = self._protocol
            # Connection may already be broken; suppress close errors
            with contextlib.suppress(Exception):
                proto.close()
            # Schedule a bounded drain so close() can observe and await
            # the reader task's teardown even though _protocol is about
            # to be nulled below. Only scheduled when a running loop is
            # available — some callers (tests, inline error paths) run
            # _invalidate outside a loop; the drain is best-effort.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:

                async def _bounded_drain() -> None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(proto.wait_closed(), timeout=self._close_timeout)

                # Strong-ref on self so the task is not GC'd before
                # close() awaits it.
                self._pending_drain = loop.create_task(_bounded_drain())
        self._protocol = None
        self._db_id = None
        self._in_use = False
        # Atomic clear of transaction-state flags: an external
        # invalidation (heartbeat path, KeyboardInterrupt mid-yield,
        # ``call_soon_threadsafe``) lands here without going through
        # ``transaction()``'s ``finally`` clause, leaving stale
        # ``_in_transaction=True`` and ``_tx_owner=<dead task>`` flags
        # that ``_check_in_use`` reads. The next user (typically from
        # the pool) sees a misleading "owned by another task" rejection.
        # ``transaction()``'s finally is idempotent on the happy path;
        # this clear is load-bearing on the external-invalidation path.
        self._in_transaction = False
        self._tx_owner = None
        # Mirror the clear for the savepoint stack tracker; the server
        # state is gone, so any stale stack entries would lie about
        # nesting on the next caller.
        self._savepoint_stack.clear()
        self._savepoint_implicit_begin = False
        self._has_untracked_savepoint = False
        # Preserve the FIRST cause: ``_ensure_connected`` raises a
        # synthetic ``DqliteConnectionError("Not connected")`` chained
        # from ``self._invalidation_cause`` whenever an
        # already-invalidated connection is touched. ``_run_protocol``
        # then catches that synthetic wrapper and calls
        # ``_invalidate(synthetic_wrapper)``, which would overwrite
        # ``_invalidation_cause`` with the wrapper itself. After several
        # invalidated calls the stored cause becomes a self-chaining
        # "Not connected → Not connected → ..." stack with the real
        # transport error buried or dropped from the surface chain. The
        # ``cause is None`` guard preserves the original root cause for
        # operators triaging a leader flip.
        if cause is not None and self._invalidation_cause is None:
            self._invalidation_cause = cause

    @staticmethod
    def _validate_params(params: Sequence[Any] | None) -> None:
        """Reject non-sequence / scalar-iterable param containers.

        The qmark paramstyle wants an ordered sequence of positional
        binds. Five shapes are actively dangerous if allowed through
        (``execute("?", <shape>)``):

        * ``str`` / ``bytes`` — iterate as single chars/bytes, silently
          binding N scalars where the caller meant one value.
        * ``bytearray`` / ``memoryview`` — same shape as ``bytes`` but
          writable; same single-byte-per-bind footgun.
        * ``Mapping`` — insertion-ordered in CPython 3.7+, but the
          qmark paramstyle is positional, not named.
        * ``set`` / ``frozenset`` — iterate in unordered fashion, so
          bindings vary across Python runs.

        Previously this validator only rejected ``str | bytes``,
        letting the three remaining shapes silently scramble bindings
        at the bind layer. Match the richer dbapi-layer check
        (``_reject_non_sequence_params``) so callers going direct to
        the client layer get the same safety net.
        """
        if params is None:
            return
        # Use ``DataError`` (a DqliteError subclass) so the client
        # contract "every error is a DqliteError" holds. Callers
        # catching ``except TypeError`` previously saw a bare
        # TypeError leak past the DqliteError hierarchy.
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

    async def _run_protocol[T](self, fn: Callable[[DqliteProtocol, int], Awaitable[T]]) -> T:
        """Run a protocol operation with standard error handling.

        Handles connection guards (_check_in_use, _ensure_connected, _in_use),
        invalidates the connection on fatal errors, and resets _in_use in all cases.
        """
        self._check_in_use()
        protocol, db_id = self._ensure_connected()
        self._in_use = True
        try:
            return await fn(protocol, db_id)
        except _WireEncodeError as e:
            # Client-side parameter-encoding error. The wire bytes were
            # never written, so the connection is still usable — convert
            # into the client-level DataError and let it propagate.
            raise DataError(str(e)) from e
        except (DqliteConnectionError, ProtocolError) as e:
            self._invalidate(e)
            raise
        except OperationalError as e:
            if e.code in LEADER_ERROR_CODES:
                self._invalidate(e)
            elif _primary_sqlite_code(e.code) in _TX_AUTO_ROLLBACK_PRIMARY_CODES:
                # Server-side SQLite engine auto-rolled-back the
                # transaction; the connection itself remains healthy.
                # Clear the local tx flags so ``in_transaction`` does
                # not lie to downstream layers (PEP 249 dbapi, SA
                # dialect) and so the next statement does not run
                # under a stale "we're still inside the user's tx"
                # assumption. Also clear the savepoint stack and the
                # autobegin flag — server-side auto-rollback discards
                # all savepoints in the rolled-back tx, mirroring the
                # cleanup discipline of ``_invalidate`` and ``close``.
                self._in_transaction = False
                self._tx_owner = None
                self._savepoint_stack.clear()
                self._savepoint_implicit_begin = False
                self._has_untracked_savepoint = False
            elif _primary_sqlite_code(e.code) == _SQLITE_BUSY and any(
                frag in (getattr(e, "raw_message", None) or e.message or "").lower()
                for frag in _RAFT_BUSY_MESSAGE_FRAGMENTS
            ):
                # SQLITE_BUSY (5) has two distinct origins in dqlite:
                # SQLite-engine-side BUSY (the user can retry the
                # failing statement and continue the same tx) AND
                # Raft-side BUSY translated by the server's gateway
                # (the in-flight write was not accepted; the server-
                # side tx may already be gone). The two cases
                # surface with the SAME primary code; the only
                # reliable distinguisher is the server's message
                # text. The "checkpoint in progress" wording emitted
                # by upstream ``dqlite-upstream/src/gateway.c`` for
                # checkpoint-contention BUSY is the one Raft-side
                # case where we know the tx-state-clear is safe.
                # Other Raft-BUSY paths (RAFT_BUSY → bare
                # ``sqlite3_errstr(SQLITE_BUSY)`` = "database is
                # locked") are indistinguishable from engine-BUSY
                # at the Python layer; users must retry explicitly.
                self._in_transaction = False
                self._tx_owner = None
                self._savepoint_stack.clear()
                self._savepoint_implicit_begin = False
                self._has_untracked_savepoint = False
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit) as e:
            # Interrupted mid-operation; we don't know how much of the
            # request/response round-trip completed, so the wire state is
            # unsafe to reuse. Invalidate and re-raise.
            self._invalidate(e)
            raise
        finally:
            self._in_use = False

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> tuple[int, int]:
        """Execute a SQL statement.

        Returns (last_insert_id, rows_affected).

        Sniffs the SQL prefix after a successful exec to keep the
        ``_in_transaction`` flag in sync with raw ``BEGIN`` / ``COMMIT``
        / ``ROLLBACK`` statements. The ``transaction()`` context
        manager updates these flags eagerly; the sniff makes the same
        invariant hold for the stdlib ``sqlite3``-style idiom of
        ``cursor.execute("BEGIN")`` ... ``cursor.execute("COMMIT")``.
        Without the sniff, ``in_transaction`` would lie about the
        engine state for raw-tx users.
        """
        self._validate_params(params)
        try:
            result = await self._run_protocol(lambda p, db: p.exec_sql(db, sql, params))
        except OperationalError as e:
            # Split the SQL once so both the deferred-FK check and the
            # conservative-flag set can consult the trailing piece for
            # multi-statement input.
            pieces: list[str] | None = None
            trigger_stmt = sql
            is_multi_with_tx_verb = False
            if ";" in sql:
                pieces = _split_top_level_statements(sql)
                if pieces:
                    trigger_stmt = pieces[-1]
                if len(pieces) > 1 and any(_starts_with_tx_verb(p) for p in pieces):
                    is_multi_with_tx_verb = True

            # Deferred-foreign-key auto-rollback: per SQLite spec
            # (https://www.sqlite.org/lang_savepoint.html), an attempt
            # to RELEASE the OUTERMOST savepoint after a deferred-FK
            # error is treated as COMMIT — the engine rolls back the
            # entire transaction. Same applies to plain COMMIT under
            # PRAGMA defer_foreign_keys=ON. SQLITE_CONSTRAINT (primary
            # 19) is NOT in ``_TX_AUTO_ROLLBACK_PRIMARY_CODES``
            # (a CHECK violation on a plain INSERT does NOT auto-
            # rollback). Verb-condition the clear: only fires when the
            # last attempted piece is plain COMMIT/END or a RELEASE of
            # the OUTERMOST frame on the savepoint stack. Using the
            # trailing piece here covers multi-statement EXEC where
            # the failing piece is the last attempted one (the dqlite
            # gateway stops on first failure).
            deferred_fk_cleared = False
            if (
                _primary_sqlite_code(e.code) == 19  # SQLITE_CONSTRAINT
                and self._sql_is_outermost_release_or_commit(trigger_stmt)
            ):
                self._in_transaction = False
                self._tx_owner = None
                self._savepoint_stack.clear()
                self._savepoint_implicit_begin = False
                self._has_untracked_savepoint = False
                deferred_fk_cleared = True

            # Multi-statement EXEC partial failure: dqlite's gateway
            # iterates the statement list, so an early statement (e.g.,
            # ``BEGIN; SAVEPOINT a; INSERT ...``) may have committed
            # server-side state before the failing statement. The
            # tracker's success-only ``_update_tx_flags_from_sql`` call
            # below is skipped on raise — leaving the local view out of
            # sync with the server's open transaction.
            #
            # Conservatively flag the connection as carrying untracked
            # state when the SQL contains a transaction-control verb
            # AND a top-level ``;`` boundary. Skip the flag-set when
            # ``_run_protocol`` already cleared the auto-rollback
            # state (either via ``_TX_AUTO_ROLLBACK_PRIMARY_CODES``
            # which clears ``_has_untracked_savepoint=False``, or via
            # the deferred-FK branch above). Otherwise we'd re-set
            # the flag and trigger a redundant pool-reset ROLLBACK.
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
        """True if ``sql`` is a single statement that is either:

        * ``COMMIT [TRANSACTION]`` / ``END [TRANSACTION]``, or
        * ``RELEASE [SAVEPOINT] <name>`` where ``<name>`` matches the
          OUTERMOST frame on the savepoint stack.

        Used to detect the deferred-foreign-key auto-rollback case
        (https://www.sqlite.org/lang_savepoint.html). The deferred-FK
        check fires at COMMIT-time or at RELEASE-of-outermost-SAVEPOINT;
        on failure SQLite tears down the entire transaction. The
        Python tracker must mirror that.
        """
        head = _strip_leading_comments(sql)
        if not head:
            return False
        upper = head.upper()
        # Plain COMMIT or END (with optional TRANSACTION keyword).
        # Gate on ``_in_transaction`` for symmetry with the RELEASE
        # branch's stack precondition: a code-19 reply to a COMMIT
        # outside an active tx is implausible (the server emits
        # SQLITE_ERROR "no transaction is active" instead), so
        # claiming the deferred-FK auto-rollback fired here would
        # zero state without a real precondition.
        if upper.startswith("COMMIT") and _is_keyword_boundary(upper, len("COMMIT")):
            return self._in_transaction
        if upper.startswith("END") and _is_keyword_boundary(upper, len("END")):
            return self._in_transaction
        # RELEASE [SAVEPOINT] <outermost>.
        if upper.startswith("RELEASE") and _is_keyword_boundary(upper, len("RELEASE")):
            name = _parse_release_name(head[len("RELEASE") :])
            if name is not None and self._savepoint_stack and name == self._savepoint_stack[0]:
                return True
        return False

    def _update_tx_flags_from_sql(self, sql: str) -> None:
        """Update _in_transaction / _tx_owner / savepoint stack after
        a successful execute, based on the leading verb of the SQL.

        The check is a cheap prefix sniff — comparable to stdlib
        ``sqlite3.Connection``'s autocommit logic — not a full SQL
        parser. It correctly tracks the common shapes (BEGIN /
        BEGIN TRANSACTION / BEGIN DEFERRED / BEGIN IMMEDIATE /
        BEGIN EXCLUSIVE / COMMIT / END / ROLLBACK) but cannot detect
        every embedded transaction-control statement (e.g. multi-
        statement strings).

        SAVEPOINT and RELEASE are tracked because SQLite's bare
        ``SAVEPOINT name`` outside an active transaction triggers an
        implicit BEGIN: the savepoint becomes the outer frame, and
        the matching ``RELEASE [SAVEPOINT] name`` ends the
        autobegun transaction. Stdlib ``sqlite3.Connection.in_transaction``
        reports ``True`` between those two points; this driver mirrors
        that semantics.

        SAVEPOINTs nested inside an explicit ``BEGIN`` do not change
        the outer transaction boundary (per SQLite spec); the tracker
        still maintains the stack so a later ROLLBACK / COMMIT can
        clear it, but ``_in_transaction`` stays True until the
        explicit COMMIT / ROLLBACK.

        ``ROLLBACK TO [SAVEPOINT] name`` leaves ``name`` active per
        SQLite spec; frames above ``name`` are popped, and
        ``_in_transaction`` is unchanged.

        Multi-statement EXEC: the dqlite server iterates the statement
        list (see ``gateway.c`` ``handle_exec_sql_done_cb``). When the
        SQL contains a top-level ``;``, split and recurse so each piece
        gets its own classification — otherwise ``execute("SAVEPOINT a;
        SAVEPOINT b;")`` would push only ``a`` locally while the server
        pushes both, desyncing the tracker.
        """
        # Cheap fast path: no semicolon, no need to walk the splitter.
        if ";" in sql:
            pieces = _split_top_level_statements(sql)
            if len(pieces) > 1:
                for piece in pieces:
                    self._update_tx_flags_from_sql(piece)
                return
            # len <= 1: either zero pieces (whitespace / comments only)
            # or a single piece whose tail ``;`` was stripped — fall
            # through to the single-statement classifier below.
            sql = pieces[0] if pieces else ""
        # Strip leading SQL comments and whitespace so the prefix
        # sniff sees past annotations like ``/* xact id */ BEGIN`` or
        # ``-- comment\nSAVEPOINT sp``. Without this, a comment-prefixed
        # transaction-control statement runs server-side but the local
        # tracker never sees the keyword and silently drifts.
        head = _strip_leading_comments(sql)
        if not head:
            return
        upper = head.upper()
        if upper.startswith("BEGIN") and _is_keyword_boundary(upper, len("BEGIN")):
            if not self._in_transaction:
                self._in_transaction = True
                # Deliberately leave ``_tx_owner`` as None for a raw
                # BEGIN: the dbapi's sync ``_run_sync`` submits each
                # call as a fresh task on the background loop, so
                # binding ``_tx_owner`` to the BEGIN-task would cause
                # the next sync ``execute`` call to be rejected as
                # "owned by another task". The async ``transaction()``
                # context manager (which DOES set ``_tx_owner``) keeps
                # the cross-task guard for its scope; raw BEGIN trusts
                # the caller to serialise their own access (matches
                # stdlib ``sqlite3`` semantics).
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
                    # SQLite implicit-begin: the savepoint is the
                    # outer frame of the new transaction. Mirror
                    # stdlib sqlite3's ``in_transaction = True``
                    # reporting. Leave ``_tx_owner = None`` for the
                    # same reason as bare BEGIN.
                    #
                    # Skip the implicit-begin transition when an outer
                    # untracked SAVEPOINT is already in flight: the
                    # server's autobegin happened on that outer frame,
                    # not on this inner tracked one. Claiming ownership
                    # here would let a subsequent RELEASE of this
                    # tracked frame flip ``_in_transaction=False``
                    # while the server still holds the autobegun tx,
                    # producing a within-checkout in-task lie about
                    # ``in_transaction``.
                    self._in_transaction = True
                    self._savepoint_implicit_begin = True
            else:
                # Quoted / backtick / square-bracket / unicode /
                # leading-digit identifier — the parser deliberately
                # returns None to avoid the case-sensitivity desync
                # described on ``_parse_savepoint_name``. The server
                # still creates the savepoint (and auto-begins a
                # transaction if none was active), so the pool-reset
                # predicate must observe the side effect via
                # ``_has_untracked_savepoint`` even though the local
                # stack stays empty.
                self._has_untracked_savepoint = True
            return
        if (
            upper.startswith("RELEASE")
            and len(upper) > len("RELEASE")
            and _is_keyword_boundary(upper, len("RELEASE"))
        ):
            name = _parse_release_name(head[len("RELEASE") :])
            if name is None:
                # Parser-rejected (quoted/backtick/bracketed/unicode/
                # leading-digit) name. The server's RELEASE pops the
                # named savepoint AND every frame above it. We don't
                # know where the named savepoint sits on the server,
                # so we don't know how many tracked frames to pop. Two
                # things follow:
                #
                # 1. Conservative-clear ``_savepoint_stack`` — any
                #    tracked frame may already be gone server-side. A
                #    later RELEASE/ROLLBACK TO of a still-tracked name
                #    would otherwise index into a ghost frame and the
                #    server would raise "no such savepoint" with no
                #    obvious correspondence in the user's SQL.
                # 2. Lock ``_has_untracked_savepoint=True`` so the
                #    pool-reset predicate keeps firing — the server
                #    may still hold outer untracked frames or an
                #    autobegun transaction we cannot model.
                self._savepoint_stack.clear()
                self._has_untracked_savepoint = True
                return
            if name in self._savepoint_stack:
                # Pop everything down to and including this name —
                # SQLite RELEASE removes the named savepoint and any
                # frames above it. Per SQLite's documentation
                # (https://www.sqlite.org/lang_savepoint.html) "the
                # name of a savepoint need not be unique. If multiple
                # savepoints have the same name, then SQLite uses the
                # most recently created savepoint with the matching
                # name." Reverse-search the stack so the LIFO contract
                # holds for duplicate names.
                idx = len(self._savepoint_stack) - 1 - self._savepoint_stack[::-1].index(name)
                del self._savepoint_stack[idx:]
                # If the stack is now empty AND the outer SAVEPOINT
                # was an autobegin, the implicit transaction ends.
                if not self._savepoint_stack and self._savepoint_implicit_begin:
                    self._in_transaction = False
                    self._tx_owner = None
                    self._savepoint_implicit_begin = False
            else:
                # The parsed name is valid bare-ASCII but does not
                # appear in the local stack — the server must have
                # created the frame via a path the tracker did not
                # observe (an earlier untracked-name SAVEPOINT, a
                # multi-statement batch the splitter could not see,
                # etc.). The server's success-only call here means
                # the named SP and every frame above it are gone
                # server-side; we don't know which (if any) of our
                # tracked frames sat above the unobserved target.
                # Mirror the parser-rejected branch: conservatively
                # clear the local stack and lock
                # ``_has_untracked_savepoint=True`` so pool reset
                # fires on return.
                self._savepoint_stack.clear()
                self._has_untracked_savepoint = True
            return
        if upper.startswith("ROLLBACK"):
            # Use bare ``lstrip()`` to consume any whitespace SQLite's
            # tokenizer accepts between keywords (space, tab, newline,
            # carriage return, form-feed). The previous ``lstrip(" ;")``
            # missed tab / CR / FF and would misclassify
            # ``ROLLBACK\tTO sp`` as a full ROLLBACK. ``;`` cannot
            # legitimately appear here because ``_split_top_level_statements``
            # has already split on top-level ``;``.
            after_upper = upper[len("ROLLBACK") :].lstrip()
            after_orig = head[len("ROLLBACK") :].lstrip()
            # SQLite grammar: ROLLBACK [TRANSACTION] [TO [SAVEPOINT]
            # name]. The TRANSACTION keyword is optional in BOTH the
            # full-rollback and the rollback-to-savepoint forms; strip
            # it before testing for ``TO`` so ``ROLLBACK TRANSACTION
            # TO SAVEPOINT sp`` is correctly classified as a savepoint
            # rollback, not a full ROLLBACK.
            tx_kw_len = len("TRANSACTION")
            if after_upper[:tx_kw_len] == "TRANSACTION" and _is_keyword_boundary(
                after_upper, tx_kw_len
            ):
                after_upper = after_upper[tx_kw_len:].lstrip()
                after_orig = after_orig[tx_kw_len:].lstrip()
            # ``ROLLBACK TO`` / ``ROLLBACK TO SAVEPOINT`` unwinds
            # frames above the named savepoint but leaves the
            # named savepoint active; the outer transaction stays
            # open.
            if after_upper.startswith("TO") and _is_keyword_boundary(after_upper, 2):
                name = _parse_release_name(after_orig[len("TO") :])
                if name is None:
                    # Parser-rejected savepoint name. Unlike RELEASE,
                    # ROLLBACK TO does NOT pop the named savepoint —
                    # it only unwinds frames ABOVE it. We can't know
                    # which (if any) of our tracked frames sit above
                    # the un-named target on the server, so we leave
                    # ``_savepoint_stack`` untouched: dropping
                    # tracked frames would over-correct in the case
                    # where every tracked frame sits BELOW the
                    # target. Lock ``_has_untracked_savepoint=True``
                    # so pool reset keeps firing — the autobegun tx
                    # (if any) is still alive on the server.
                    self._has_untracked_savepoint = True
                    return
                if name in self._savepoint_stack:
                    # Reverse-search to match the most recently created
                    # savepoint with this name (SQLite's LIFO rule for
                    # duplicate names — see RELEASE branch above).
                    idx = len(self._savepoint_stack) - 1 - self._savepoint_stack[::-1].index(name)
                    del self._savepoint_stack[idx + 1 :]
                else:
                    # The parsed name does not appear in the local
                    # stack. The server's success-only call here
                    # means the named target exists on its side and
                    # frames above it have been unwound. We don't
                    # know which (if any) of our tracked frames sat
                    # above the unobserved target; over-correcting
                    # by clearing tracked frames could drop frames
                    # that sat BELOW the target. Conservative: leave
                    # ``_savepoint_stack`` untouched and lock
                    # ``_has_untracked_savepoint=True`` so the
                    # pool-reset safety net fires.
                    self._has_untracked_savepoint = True
                return
            if self._in_transaction:
                self._in_transaction = False
                self._tx_owner = None
                self._savepoint_stack.clear()
                self._savepoint_implicit_begin = False
                self._has_untracked_savepoint = False
            else:
                # ROLLBACK without an active transaction is a no-op on
                # the server. The invariant "non-empty stack implies
                # _in_transaction" should keep the stack empty here,
                # but defensively clear all four state fields so a
                # future state-machine bug cannot let stale stack
                # entries leak through. Mirrors the discipline at
                # close() / _invalidate / the auto-rollback branch.
                self._savepoint_stack.clear()
                self._savepoint_implicit_begin = False
                self._has_untracked_savepoint = False
            return
        if upper.startswith("COMMIT") or upper.startswith("END"):
            # ``COMMIT`` / ``END`` close the outer transaction. Both
            # forms (``COMMIT``, ``COMMIT TRANSACTION``, ``END``,
            # ``END TRANSACTION``) end here.
            verb = "COMMIT" if upper.startswith("COMMIT") else "END"
            if _is_keyword_boundary(upper, len(verb)):
                if self._in_transaction:
                    self._in_transaction = False
                    self._tx_owner = None
                # The autobegin-deferral path (when an outer
                # untracked SAVEPOINT had set
                # ``_has_untracked_savepoint=True``) deliberately
                # allows ``stack`` non-empty AND
                # ``_in_transaction=False``. Clear the stack and
                # the implicit-begin flag UNCONDITIONALLY here so a
                # COMMIT that closes the server-side autobegun tx
                # does not leave a ghost frame in the local stack.
                # Mirrors the defensive double-clear the symmetric
                # ROLLBACK branch already performs in both arms.
                self._savepoint_stack.clear()
                self._savepoint_implicit_begin = False
                self._has_untracked_savepoint = False
                return

    async def query_raw(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute a query and return raw (column_names, rows).

        Unlike fetch() which returns dicts, this returns the raw tuple
        of (column_names, rows) from the wire protocol. Intended for
        DBAPI cursor implementations that need column names separately.

        See ``query_raw_typed`` when per-column wire ``ValueType`` codes
        are also needed (used by ``cursor.description``).
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))

    async def query_raw_typed(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> tuple[list[str], list[int], list[list[int]], list[list[Any]]]:
        """Execute a query and return (column_names, column_types, row_types, rows).

        ``column_types`` are per-column wire ``ValueType`` integer tags
        from the first response frame — suitable for populating DBAPI
        ``cursor.description[i][1]`` (``type_code``). ``row_types`` is
        one list of wire tags per decoded row; SQLite is dynamically
        typed, so different rows in the same column can carry
        different wire types (under UNION, ``CASE``, ``COALESCE``,
        ``typeof()``), and callers applying result-side converters
        need the per-row list rather than a collapsed first-row view.
        See ``dqlitewire.ValueType`` for the full enum. Use
        ``query_raw`` when type codes are not needed.
        """
        self._validate_params(params)
        return await self._run_protocol(lambda p, db: p.query_sql_typed(db, sql, params))

    async def fetch(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        self._validate_params(params)
        columns, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        return [dict(zip(columns, row, strict=True)) for row in rows]

    async def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[list[Any]]:
        """Execute a query and return results as list of lists."""
        self._validate_params(params)
        _, rows = await self._run_protocol(lambda p, db: p.query_sql(db, sql, params))
        return rows

    async def fetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> dict[str, Any] | None:
        """Execute a query and return the first result.

        Note: dqlite returns all matching rows over the wire. For large
        result sets, add ``LIMIT 1`` to your query to avoid excessive
        memory usage.
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
        """Context manager for transactions.

        Issues a bare ``BEGIN`` (SQLite default ``BEGIN DEFERRED``) on
        entry, ``COMMIT`` on clean exit, and ``ROLLBACK`` if the body
        raises. dqlite's Raft FSM serializes transactions across the
        cluster regardless of the ``DEFERRED`` / ``IMMEDIATE`` /
        ``EXCLUSIVE`` qualifier — isolation is always SERIALIZABLE —
        so the qualifier choice has no semantic effect on dqlite. The
        ``BEGIN`` literal matches the C/Go peer behaviour.

        Cancellation contract:
        - Cancellation during BEGIN: state cleared, CancelledError
          propagates.
        - Cancellation during the body: ROLLBACK is attempted. If
          ROLLBACK itself is cancelled, the connection is invalidated
          and CancelledError propagates (structured-concurrency
          contract — TaskGroup / asyncio.timeout() require this).
        - Cancellation during COMMIT: connection invalidated
          (server-side state ambiguous), CancelledError propagates.
        - Cancellation during ROLLBACK (body already raised): connection
          invalidated, CancelledError propagates and supersedes the
          body exception (Python chains it via ``__context__``).

        Non-cancellation ROLLBACK failure: connection is
        invalidated so the pool discards it instead of reusing a
        Python-side "_in_transaction=False" connection with live
        server-side transaction state.

        Shielding caveat: ``asyncio.shield(conn.execute(...))`` inside
        the body creates a new task whose ``current_task()`` is not
        ``_tx_owner``. ``_check_in_use`` rejects the shielded call
        with ``InterfaceError("owned by another task")``. Shield the
        **entire** ``async with conn.transaction():`` block if you
        need defensive-rollback semantics against outer cancellation
        — not individual operations inside it.
        """
        # An untracked SAVEPOINT (parser-rejected name issued without a
        # preceding BEGIN) auto-begins a server-side tx without flipping
        # ``_in_transaction``. Surface a dedicated diagnostic so the
        # user sees the SAVEPOINT root cause rather than a wire-level
        # "cannot start a transaction within a transaction" from BEGIN.
        # ``_tx_owner`` is None in this case, so the owned-by-another-
        # task branch below would render a misleading
        # ``repr(None)`` owner.
        if self._has_untracked_savepoint and not self._in_transaction:
            raise InterfaceError(
                "Cannot start transaction: a SAVEPOINT outside an explicit "
                "BEGIN is currently open on this connection (the SQLite "
                "engine has auto-begun a transaction). Issue COMMIT / "
                "ROLLBACK or RELEASE the outer SAVEPOINT before entering "
                "transaction()."
            )
        # Only the SAME task re-entering an open transaction is "nested".
        # A sibling task hitting an in-progress transaction should see
        # the "owned by another task" diagnostic so the actual remedy —
        # acquire a separate connection from the pool — is obvious. The
        # nested-tx message historically fired for sibling-task usage
        # too and pointed users at SAVEPOINT, which was the wrong
        # guidance for that pattern.
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

        # Set the flags before the BEGIN await — the early set is a
        # secondary guard atop ``_run_protocol``'s ``_in_use`` flag.
        # While ``_in_use`` rejects concurrent calls with "another
        # operation in progress", the early ``_in_transaction=True``
        # makes a *task switch mid-BEGIN* surface as the more specific
        # "nested transactions are not supported" — useful when
        # callers wrap operations in a single shared connection by
        # mistake. The except clause clears on any failure path.
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
                # COMMIT was sent but failed. Server-side state is ambiguous
                # (maybe committed, maybe still open, maybe rolled back). We
                # cannot safely reuse this connection — invalidate so the
                # pool discards it instead of recycling an unknown state.
                # Pass the in-flight exception as the cause so subsequent
                # ``_ensure_connected`` raises chain back to the cancel /
                # OperationalError that triggered the invalidation, instead
                # of dropping ``__cause__`` on the floor.
                self._invalidate(exc)
            else:
                # Body raised before COMMIT; try to roll back.
                #
                # Narrow suppression to Exception (NOT BaseException):
                # CancelledError / KeyboardInterrupt / SystemExit must
                # propagate. Previously ``suppress(BaseException)``
                # swallowed cancellation, breaking structured-concurrency
                # contracts.
                #
                # If ROLLBACK fails for any reason (including the narrow
                # cancellation catch below), the connection's transaction
                # state is unknowable from our side and the connection
                # must be invalidated so the pool discards it on return
                # from our side. The original body exception is still the
                # one that propagates, except for cancellation which
                # takes precedence.
                try:
                    await self.execute(_TRANSACTION_ROLLBACK_SQL)
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    # Rollback interrupted mid-flight. Server-side tx is
                    # in an unknown state; invalidate and propagate the
                    # higher-priority signal.
                    logger.debug(
                        "transaction(address=%s, id=%s): rollback was "
                        "cancelled mid-flight; connection invalidated; "
                        "propagating cancellation",
                        self._address,
                        id(self),
                        exc_info=True,
                    )
                    self._invalidate()
                    raise
                except OperationalError as roll_exc:
                    # SQLITE_ERROR with the "no transaction is active"
                    # wording is the deterministic "nothing to roll
                    # back" reply — the body exception aborted before
                    # the implicit BEGIN reached the server, or the
                    # server already auto-rolled-back. The connection
                    # is healthy; preserve it and re-raise the body
                    # exception. Mirrors the dbapi layer's
                    # _NO_TX_CODES whitelist (ISSUE-696).
                    if _is_no_tx_rollback_error(roll_exc):
                        logger.debug(
                            "transaction(address=%s, id=%s): rollback "
                            "found no active transaction (server-side "
                            "tx already gone); preserving connection",
                            self._address,
                            id(self),
                        )
                        # Server reports no transaction is active — the
                        # savepoint stack is necessarily gone too. Mirror
                        # the all-clear discipline enforced at
                        # _invalidate / close / _run_protocol's
                        # auto-rollback branch so the pool-reset
                        # predicate doesn't see a stale stack and
                        # re-issue another (also benign) ROLLBACK.
                        self._savepoint_stack.clear()
                        self._savepoint_implicit_begin = False
                        self._has_untracked_savepoint = False
                    else:
                        logger.debug(
                            "transaction(address=%s, id=%s): rollback failed "
                            "with OperationalError; connection invalidated; "
                            "propagating original body exception",
                            self._address,
                            id(self),
                            exc_info=True,
                        )
                        # Pass the rollback failure as the cause so the
                        # next "Not connected" diagnostic chains to it
                        # via __cause__. Mirrors the commit-attempted
                        # arm above (line ~1545) which already passes
                        # ``exc``. The body exception still propagates
                        # via ``raise`` below — this only affects the
                        # diagnostic visible on a SUBSEQUENT call to
                        # the invalidated connection.
                        self._invalidate(roll_exc)
                except Exception as roll_exc:
                    # Rollback failed for a non-OperationalError reason.
                    # Invalidate so the pool discards on return, then
                    # re-raise the ORIGINAL body exception (below)
                    # — rollback failure is a secondary concern.
                    logger.debug(
                        "transaction(address=%s, id=%s): rollback failed; "
                        "connection invalidated; propagating original body "
                        "exception",
                        self._address,
                        id(self),
                        exc_info=True,
                    )
                    self._invalidate(roll_exc)
            raise
        finally:
            # Defence-in-depth: the success path's COMMIT and the
            # failure-path branches above all clear the four tx fields
            # before reaching here, but clearing again here keeps the
            # invariant local to transaction()'s exit so a future
            # refactor that splits COMMIT from state-update cannot
            # silently regress. Idempotent — already-cleared fields stay
            # cleared.
            self._tx_owner = None
            self._in_transaction = False
            self._savepoint_stack.clear()
            self._savepoint_implicit_begin = False
            self._has_untracked_savepoint = False
