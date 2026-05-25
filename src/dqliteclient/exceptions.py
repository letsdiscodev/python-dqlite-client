"""Exceptions for dqlite client."""

from typing import Any, ClassVar

from dqlitewire import DEFAULT_MAX_RAW_MESSAGE as _DEFAULT_MAX_RAW_MESSAGE
from dqlitewire import ProtocolError as _WireProtocolError
from dqlitewire import cap_raw_message as _wire_cap_raw_message
from dqlitewire import sanitize_server_text as _sanitize_server_text

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
    server-supplied diagnostic, un-suffixed (but capped at
    ``_MAX_RAW_MESSAGE`` codepoints to bound cross-process pickled
    exception graphs) — so callers (the dbapi-layer classifier, SA's
    ``is_disconnect``) can read forensic-grade text without falling
    back to ``str(e)`` (which can be truncated by display caps or
    padded by per-class wrap prefixes). Defaults to ``None`` for
    raises that are purely client-side (no server text in scope).

    The ``OperationalError`` subclass adds an extra display-message
    cap on top (``_MAX_DISPLAY_MESSAGE`` — a stricter bound on the
    user-facing string than ``raw_message``); all other subclasses
    inherit this base constructor unchanged and get the
    ``_MAX_RAW_MESSAGE`` cap automatically.
    """

    # Cap on ``raw_message``. The rationale (~64 KiB
    # ``FailureResponse``, ``BaseExceptionGroup`` fan-out under hostile
    # peers, cross-process pickling) lives in
    # ``dqlitewire.DEFAULT_MAX_RAW_MESSAGE`` — the single source of
    # truth shared with ``dqlitedbapi``. Subclasses may override.
    _MAX_RAW_MESSAGE: ClassVar[int] = _DEFAULT_MAX_RAW_MESSAGE

    raw_message: str | None

    def __init__(self, *args: object, raw_message: str | None = None) -> None:
        super().__init__(*args)
        self.raw_message = self._cap_raw_message(raw_message)

    @classmethod
    def _cap_raw_message(cls, raw_message: str | None) -> str | None:
        # Thin wrapper over the wire-layer helper so the truncation
        # logic + suffix wording lives in one place. The classmethod
        # shape is preserved so subclasses can override
        # ``_MAX_RAW_MESSAGE`` at the class level.
        return _wire_cap_raw_message(raw_message, cls._MAX_RAW_MESSAGE)

    def __getstate__(self) -> dict[str, object]:
        """State capture for pickle.

        Default ``Exception.__reduce__`` returns ``(cls, self.args)``
        which reconstructs via ``cls(*args)``. That loses every
        field set on the instance after ``Exception.__init__`` —
        most notably ``raw_message`` (set on the ``DqliteError``
        base) and the ``code`` carried by ``DqliteConnectionError``.
        The carriers exist precisely so the wire-level signal
        survives cross-process pickling (``ProcessPoolExecutor``,
        ``multiprocessing.Queue``, Celery task results, SA's
        multiprocess pool).

        Pair with :meth:`__setstate__` so the explicit
        ``__getstate__`` / ``__setstate__`` shape replaces the
        previous ``__reduce__``-with-dict-overlay. The new shape
        makes the captured state visible in stack traces (no longer
        hidden behind ``self.__dict__.copy()``) and tolerates
        future subclass refactors that add slot-based attributes
        (those slot values can be added to the dict explicitly in
        the subclass override rather than silently lost behind
        ``__dict__``).
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any] | None) -> None:
        if state:
            self.__dict__.update(state)

    def __reduce__(
        self,
    ) -> tuple[type["DqliteError"], tuple[object, ...], dict[str, object]]:
        # 3-tuple form: pickle reconstructs via ``cls(*args)`` then
        # invokes ``__setstate__`` with the dict from
        # ``__getstate__``. Subclasses inherit this discipline
        # automatically — every subclass we ship today carries its
        # extra fields on ``__dict__`` (no ``__slots__``), so the
        # default ``__getstate__`` returning ``self.__dict__.copy()``
        # captures them.
        return (self.__class__, self.args, self.__getstate__())


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

    def __repr__(self) -> str:
        # Surface ``code`` in ``repr(exc)`` so forensic log sites
        # (``logger.exception``, ``logger.error("%r", exc)``,
        # ``pytest``'s exception-with-args display) capture the
        # wire-level code rather than dropping it. Mirrors the
        # dbapi sibling at
        # ``dqlitedbapi.exceptions.DatabaseError.__repr__``;
        # asymmetry meant a caller catching ``DqliteConnectionError``
        # at the client surface logged less than a caller catching
        # the dbapi-layer equivalent. ``code=None`` (transport-only
        # faults with no wire-level signal) renders without the
        # ``code=`` suffix to avoid a noisy ``code=None`` line.
        msg = self.args[0] if self.args else ""
        cls = type(self).__name__
        if self.code is None:
            return f"{cls}({msg!r})"
        return f"{cls}({msg!r}, code={self.code!r})"


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

    ``self.message`` is also display-safe: control bytes (CR, NUL,
    ANSI escape, etc.) are neutralised by ``sanitize_server_text``
    at construction so a third-party caller raising
    ``OperationalError(server_text, code)`` directly cannot leak
    log-injection vectors into ``str(e)``. LF / Tab are intentionally
    preserved so multi-line server diagnostics render correctly;
    ``sanitize_for_log`` is the log-callsite helper that additionally
    escapes those. ``raw_message`` carries the verbatim peer text
    untouched for forensic recovery.
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
        """Positional shape: ``(message, code)``. Aligned with
        ``dqlitedbapi.exceptions.OperationalError.__init__`` so cross-
        package bridges (SA dialect ``is_disconnect``, retry middleware,
        custom decorators) can pass args positionally without silently
        swapping fields. Mirrors stdlib ``sqlite3.Error.__init__(msg, *)``
        single-message convention.
        """
        self.code = code
        # ``raw_message`` is the verbatim server text. Callers compose
        # the display ``message`` with peer-address suffix / "Failed to
        # connect:" prefix etc. and pass the unadorned server text as
        # ``raw_message=`` so the contract that raw_message is the
        # verbatim server text is preserved through the dbapi-layer
        # plumbing. Old call sites that omit the kwarg still get the
        # previous behaviour (``raw_message`` defaults to ``message``).
        # The ~4 KiB cap on raw_message is applied by ``DqliteError``.
        resolved_raw_message = message if raw_message is None else raw_message
        # Display sanitisation: neutralise control / bidi / invisible
        # bytes in the display field before truncation so the
        # invariant "``self.message`` is always display-safe" holds at
        # the class boundary, not at every call site. Every first-
        # party call site that builds an ``OperationalError`` from a
        # ``FailureResponse.message`` already pre-sanitises through
        # ``_sanitize_display_text`` (an alias for the same wire-side
        # helper); applying it again is idempotent. The defence-in-
        # depth catches future call sites that forget the helper, and
        # any third-party caller raising
        # ``OperationalError(server_text, code)`` directly. LF / Tab
        # are intentionally preserved; ``sanitize_for_log`` is the
        # log-callsite helper that additionally escapes those.
        sanitised = _sanitize_server_text(message)
        if len(sanitised) > self._MAX_DISPLAY_MESSAGE:
            # ``len(sanitised)`` and the slice cap count Python codepoints,
            # not UTF-8 bytes. Match the unit in the marker so an
            # operator inspecting a truncated message can compute the
            # original size without converting between units. The
            # ``overflow`` count is server-controlled when the message
            # originates from a peer ``FailureResponse`` — a deliberate
            # forensic-vs-disclosure trade-off (operators want the
            # original size for triage; the marginal info-disclosure of
            # exposing the size class outweighs the loss).
            overflow = len(sanitised) - self._MAX_DISPLAY_MESSAGE
            self.message = (
                f"{sanitised[: self._MAX_DISPLAY_MESSAGE]}... [truncated, {overflow} codepoints]"
            )
        else:
            self.message = sanitised
        # Pass the TRUNCATED display ``self.message`` and ``code``
        # through as args so ``self.args == (truncated_message, code)``;
        # pickle / deepcopy reconstruct via
        # ``OperationalError(truncated_message, code, raw_message=...)``.
        # Without truncating ``args`` here, the pickled payload would
        # carry the original un-truncated ``message`` argument —
        # defeating the bound for cross-process exception graphs.
        # ``DqliteError`` then caps ``raw_message``.
        super().__init__(self.message, code, raw_message=resolved_raw_message)

    def __str__(self) -> str:
        # Plain message — matches stdlib ``sqlite3.OperationalError("foo")``
        # whose ``str(e) == "foo"`` even when ``sqlite_errorcode`` is set,
        # and matches the dbapi-layer wrapper classes which also do not
        # prefix the error code. ``code`` and ``raw_message`` remain
        # available via attribute access and ``__repr__`` for callers
        # that want structured access; logging code that wants both can
        # format ``f"{e!r}"`` or read the attributes directly.
        return self.message

    def __repr__(self) -> str:
        # Surface ``code`` in the labelled-form so forensic log sites
        # (``logger.exception``, ``logger.error("%r", exc)``, pytest's
        # exception-with-args display) capture the wire-level code
        # rather than rendering the inherited bare ``(message, code)``
        # tuple form. Mirrors the sibling
        # ``DqliteConnectionError.__repr__`` and the dbapi-layer
        # ``dqlitedbapi.exceptions.OperationalError`` shape — cross-
        # package forensic log tooling expects symmetric labelled
        # output regardless of whether the exception is unwrapped or
        # wrapped. ``code`` is unconditionally set on this class so
        # the suffix always renders.
        return f"{type(self).__name__}({self.message!r}, code={self.code!r})"
