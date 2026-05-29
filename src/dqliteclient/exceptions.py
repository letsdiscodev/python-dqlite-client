"""Exceptions for dqlite client."""

from typing import Any, ClassVar

from dqlitewire import DEFAULT_MAX_RAW_MESSAGE as _DEFAULT_MAX_RAW_MESSAGE
from dqlitewire import ProtocolError as _WireProtocolError
from dqlitewire import cap_raw_message as _wire_cap_raw_message
from dqlitewire import sanitize_server_text as _sanitize_server_text

__all__ = [
    "AmbiguousCommitError",
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
    """Base exception for dqlite client errors.

    ``raw_message`` carries the verbatim server diagnostic (capped, un-suffixed) for forensic
    callers; ``None`` for purely client-side raises.
    """

    # Cap on raw_message; rationale shared with dqlitedbapi via DEFAULT_MAX_RAW_MESSAGE.
    _MAX_RAW_MESSAGE: ClassVar[int] = _DEFAULT_MAX_RAW_MESSAGE

    raw_message: str | None

    def __init__(self, *args: object, raw_message: str | None = None) -> None:
        super().__init__(*args)
        self.raw_message = self._cap_raw_message(raw_message)

    @classmethod
    def _cap_raw_message(cls, raw_message: str | None) -> str | None:
        # classmethod shape lets subclasses override _MAX_RAW_MESSAGE at the class level.
        return _wire_cap_raw_message(raw_message, cls._MAX_RAW_MESSAGE)

    def __getstate__(self) -> dict[str, object]:
        # Capture instance fields (raw_message, code) that default Exception pickling drops,
        # so the wire-level signal survives cross-process pickling.
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any] | None) -> None:
        if state:
            self.__dict__.update(state)

    def __reduce__(
        self,
    ) -> tuple[type["DqliteError"], tuple[object, ...], dict[str, object]]:
        # 3-tuple form: reconstruct via cls(*args), then __setstate__ restores extra fields.
        return (self.__class__, self.args, self.__getstate__())


class DqliteConnectionError(DqliteError):
    """Error establishing or maintaining connection.

    Optional ``code`` preserves the wire-level signal across a connect-path rewrap.
    """

    code: int | None

    def __init__(
        self,
        message: str = "",
        *,
        code: int | None = None,
        raw_message: str | None = None,
    ) -> None:
        self.code = code
        super().__init__(message, raw_message=raw_message)

    def __repr__(self) -> str:
        # Surface code so forensic log sites capture it; omit when None to avoid noise.
        msg = self.args[0] if self.args else ""
        cls = type(self).__name__
        if self.code is None:
            return f"{cls}({msg!r})"
        return f"{cls}({msg!r}, code={self.code!r})"


class ProtocolError(DqliteError, _WireProtocolError):
    """Protocol-level error.

    Dual parent so both ``except DqliteError`` and ``except dqlitewire.ProtocolError`` catch it.
    """


class InterfaceError(DqliteError):
    """Misuse of the client interface (e.g. concurrent access on a connection)."""


class ClusterError(DqliteError):
    """Cluster-related error (leader not found, etc)."""


class ClusterPolicyError(ClusterError):
    """Leader redirect rejected by a configured redirect_policy.

    Deterministic (config mismatch, not transient) so it must bypass the connect-time retry loop.
    """


class DataError(DqliteError):
    """Client-side parameter-encoding error (value not serializable onto the wire)."""


class OperationalError(DqliteError):
    """Database operation error.

    ``self.message`` is sanitised and truncated to ~1 KiB for display (log-amplification
    defence); ``self.raw_message`` keeps the untruncated verbatim peer text for forensics.
    """

    _MAX_DISPLAY_MESSAGE: ClassVar[int] = 1024

    code: int
    message: str
    raw_message: str

    def __init__(
        self,
        message: str,
        code: int,
        *,
        raw_message: str | None = None,
    ) -> None:
        # Positional shape (message, code) aligned with dqlitedbapi and stdlib sqlite3.
        self.code = code
        # Defaulting raw_message to message keeps old call sites that omit the kwarg working.
        resolved_raw_message = message if raw_message is None else raw_message
        # Sanitise control/bidi/invisible bytes so self.message is display-safe at the class
        # boundary; idempotent vs first-party pre-sanitising. LF/Tab preserved.
        sanitised = _sanitize_server_text(message)
        if len(sanitised) > self._MAX_DISPLAY_MESSAGE:
            # Count codepoints (not UTF-8 bytes) and label the marker likewise so operators
            # can compute the original size.
            overflow = len(sanitised) - self._MAX_DISPLAY_MESSAGE
            self.message = (
                f"{sanitised[: self._MAX_DISPLAY_MESSAGE]}... [truncated, {overflow} codepoints]"
            )
        else:
            self.message = sanitised
        # Pass the truncated self.message as args so pickle/deepcopy stay bounded; passing the
        # original would defeat the cap for cross-process exception graphs.
        super().__init__(self.message, code, raw_message=resolved_raw_message)

    def __str__(self) -> str:
        # Plain message (no code prefix) — matches stdlib sqlite3 and the dbapi-layer wrappers.
        return self.message

    def __repr__(self) -> str:
        # Labelled form surfaces code for forensic log sites; code is always set on this class.
        return f"{type(self).__name__}({self.message!r}, code={self.code!r})"


class AmbiguousCommitError(OperationalError):
    """COMMIT mid-flight failure with genuinely unknown server-side state.

    The Raft entry may or may not have been replicated; the client cannot tell from the wire.
    Retries MUST be treated as at-least-once (use idempotent DML or an out-of-band state check).
    """
