"""Exceptions for dqlite client."""

from dqlitewire.exceptions import ProtocolError as _WireProtocolError


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
    """Database operation error."""

    code: int
    message: str

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        # Pass ``code`` and ``message`` through as separate args so
        # ``self.args == (code, message)``; otherwise ``pickle`` /
        # ``copy.deepcopy`` reconstruct via ``OperationalError(*args)``
        # with a single positional argument and raise ``TypeError``.
        super().__init__(code, message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
