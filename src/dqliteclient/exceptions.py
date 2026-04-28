"""Exceptions for dqlite client."""

from typing import ClassVar

from dqlitewire.exceptions import ProtocolError as _WireProtocolError

__all__ = [
    "ClusterError",
    "ClusterPolicyError",
    "DataError",
    "DqliteConnectionError",
    "DqliteError",
    "InterfaceError",
    "OperationalError",
    "ProtocolError",
]


class DqliteError(Exception):
    """Base exception for dqlite client errors."""


class DqliteConnectionError(DqliteError):
    """Error establishing or maintaining connection."""


class ProtocolError(DqliteError, _WireProtocolError):
    """Protocol-level error.

    Inherits from both the client's ``DqliteError`` hierarchy (so
    ``except DqliteError`` still catches it) and the wire-layer
    ``dqlitewire.exceptions.ProtocolError`` (so a caller who wants to
    catch protocol-level problems at *any* layer can write
    ``except dqlitewire.ProtocolError`` and get both the wire- and
    client-level variants). Without the dual parent, the two identically
    named classes caught only half the surface depending on which module
    the caller imported.
    """


class InterfaceError(DqliteError):
    """Misuse of the client interface (e.g. concurrent access on a connection)."""


class ClusterError(DqliteError):
    """Cluster-related error (leader not found, etc)."""


class ClusterPolicyError(ClusterError):
    """Leader redirect rejected by a configured redirect_policy.

    Subclass of :class:`ClusterError` so callers already writing
    ``except ClusterError`` continue to work. Raised by
    ``ClusterClient._check_redirect`` when the operator-supplied policy
    rejects a redirect target; the rejection is deterministic (it
    reflects a configuration mismatch, not a transient failure) and
    therefore must bypass the connect-time retry loop — retrying would
    simply re-invoke the same policy against the same store and
    multiply the wall-clock cost.
    """


class DataError(DqliteError):
    """Client-side parameter-encoding error.

    Raised when a parameter value cannot be serialized onto the wire —
    e.g. an int outside ``[-2^63, 2^63)``, an embedded null byte in a
    TEXT string, or an unsupported Python type.
    """


class OperationalError(DqliteError):
    """Database operation error.

    ``FailureResponse.message`` from the wire can be up to 64 KiB
    (sanitised but unbounded otherwise). Embedding that verbatim in
    every ``OperationalError`` inflates log lines, traceback renders,
    and any ``str(exc)`` consumer (e.g. SQLAlchemy's ``is_disconnect``
    substring scan) — a mild log-amplification vector for a hostile
    peer. Truncate to ~1 KiB on ``self.message`` for display; keep the
    untruncated string on ``self.raw_message`` for callers that need
    forensic access. Pickle / ``copy.deepcopy`` stay lossless because
    ``super().__init__`` keeps the raw payload on ``self.args``.
    """

    _MAX_DISPLAY_MESSAGE: ClassVar[int] = 1024

    code: int
    message: str
    raw_message: str

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.raw_message = message
        if len(message) > self._MAX_DISPLAY_MESSAGE:
            overflow = len(message) - self._MAX_DISPLAY_MESSAGE
            self.message = (
                f"{message[: self._MAX_DISPLAY_MESSAGE]}... [truncated, {overflow} bytes]"
            )
        else:
            self.message = message
        # Pass ``code`` and the RAW message through as separate args so
        # ``self.args == (code, raw_message)``; pickle / deepcopy
        # reconstruct via ``OperationalError(*args)`` and re-run the
        # truncation, preserving both fields losslessly.
        super().__init__(code, message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
