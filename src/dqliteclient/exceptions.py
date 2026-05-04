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
    """Base exception for dqlite client errors.

    Carries an optional ``raw_message`` attribute — the verbatim
    server-supplied diagnostic, un-truncated and un-suffixed — so
    callers (the dbapi-layer classifier, SA's ``is_disconnect``)
    can read forensic-grade text without falling back to
    ``str(e)`` (which can be truncated by display caps or padded
    by per-class wrap prefixes). Defaults to ``None`` for raises
    that are purely client-side (no server text in scope).

    The ``OperationalError`` subclass overrides ``__init__`` to keep
    its existing display-truncation semantics (a 1 KiB cap on the
    user-facing ``message`` field with the un-truncated text on
    ``raw_message``); all other subclasses inherit this base
    constructor unchanged.
    """

    raw_message: str | None

    def __init__(self, *args: object, raw_message: str | None = None) -> None:
        super().__init__(*args)
        self.raw_message = raw_message

    def __reduce__(
        self,
    ) -> tuple[type["DqliteError"], tuple[object, ...], dict[str, object]]:
        # Default ``Exception.__reduce__`` returns ``(cls, self.args)``,
        # which reconstructs via ``cls(*args)``. That loses every field
        # set on the instance after ``Exception.__init__`` — most
        # notably ``raw_message`` (set on the ``DqliteError`` base) and
        # the ``code`` carried by ``DqliteConnectionError``. The
        # carriers were added in cycle 27 (XP2 / XP3) precisely so the
        # wire-level signal would survive cross-process pickling
        # (``ProcessPoolExecutor``, ``multiprocessing.Queue``, Celery
        # task results, SA's multiprocess pool); without overriding
        # ``__reduce__`` the round-trip silently drops them.
        #
        # Return the 3-tuple ``(callable, args, state)`` form so pickle
        # reconstructs via ``cls(*args)`` then applies state via
        # ``self.__dict__.update(state)`` — preserving every attribute
        # we set on the instance (``raw_message`` on the base, ``code``
        # on subclasses, the truncated ``message`` on
        # ``OperationalError``). All subclasses of ``DqliteError``
        # inherit this discipline automatically.
        return (self.__class__, self.args, self.__dict__.copy())


class DqliteConnectionError(DqliteError):
    """Error establishing or maintaining connection.

    Optionally carries ``code`` so the wire-level signal survives a
    connect-path rewrap of an upstream ``OperationalError`` (e.g. the
    ``LEADER_ERROR_CODES`` branch in ``DqliteConnection.connect()``).
    ``raw_message`` (the verbatim server text) is inherited from the
    ``DqliteError`` base.
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
    # Cap on the un-truncated ``raw_message``. The wire layer caps
    # ``FailureResponse.message`` at ~64 KiB; combined with
    # ``BaseExceptionGroup`` chains in ``find_leader`` and
    # ``pool.initialize``, a 100-node sweep against hostile peers
    # would otherwise produce ~6 MB exception payloads that flow
    # through cross-process pickling (``ProcessPoolExecutor``,
    # Celery, structured-error capture). 4 KiB is well above any
    # realistic SQLite error string (the longest in CPython's test
    # suite is ~200 chars) while bounding the worst-case fan-out.
    _MAX_RAW_MESSAGE: ClassVar[int] = 4 * 1024

    code: int
    message: str
    raw_message: str

    def __init__(
        self,
        code: int,
        message: str,
        *,
        raw_message: str | None = None,
    ) -> None:
        self.code = code
        # ``raw_message`` is the verbatim server text (un-truncated,
        # un-suffixed). Callers compose the display ``message`` with
        # peer-address suffix / "Failed to connect:" prefix etc. and
        # pass the unadorned server text as ``raw_message=`` so the
        # contract that raw_message is the verbatim server text is
        # preserved through the dbapi-layer plumbing. Old call sites
        # that omit the kwarg still get the previous behaviour
        # (``raw_message`` defaults to ``message``).
        resolved_raw_message = message if raw_message is None else raw_message
        # Bound ``raw_message`` so cross-process pickled exception
        # graphs stay small even under hostile-peer fan-out.
        if len(resolved_raw_message) > self._MAX_RAW_MESSAGE:
            raw_overflow = len(resolved_raw_message) - self._MAX_RAW_MESSAGE
            resolved_raw_message = (
                resolved_raw_message[: self._MAX_RAW_MESSAGE]
                + f"... [raw_message truncated, {raw_overflow} codepoints]"
            )
        if len(message) > self._MAX_DISPLAY_MESSAGE:
            # ``len(message)`` and the slice cap count Python codepoints,
            # not UTF-8 bytes. Match the unit in the marker so an
            # operator inspecting a truncated message can compute the
            # original size without converting between units. The
            # ``overflow`` count is server-controlled when the message
            # originates from a peer ``FailureResponse`` — a deliberate
            # forensic-vs-disclosure trade-off (operators want the
            # original size for triage; the marginal info-disclosure of
            # exposing the size class outweighs the loss).
            overflow = len(message) - self._MAX_DISPLAY_MESSAGE
            self.message = (
                f"{message[: self._MAX_DISPLAY_MESSAGE]}... [truncated, {overflow} codepoints]"
            )
        else:
            self.message = message
        # Pass ``code`` and the display ``message`` through as separate
        # args so ``self.args == (code, message)``; pickle / deepcopy
        # reconstruct via ``OperationalError(*args)``. Pass
        # ``raw_message=`` through to the base so the ``DqliteError``
        # ``raw_message`` slot is set there (keeps a single source of
        # truth across the hierarchy).
        super().__init__(code, message, raw_message=resolved_raw_message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
