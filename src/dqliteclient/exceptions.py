"""Exceptions for dqlite client."""


class DqliteError(Exception):
    """Base exception for dqlite client errors."""

    pass


class ConnectionError(DqliteError):
    """Error establishing or maintaining connection."""

    pass


class ProtocolError(DqliteError):
    """Protocol-level error."""

    pass


class ClusterError(DqliteError):
    """Cluster-related error (leader not found, etc)."""

    pass


class OperationalError(DqliteError):
    """Database operation error."""

    code: int
    message: str

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
