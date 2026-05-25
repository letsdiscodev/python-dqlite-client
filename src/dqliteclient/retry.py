"""Retry utilities with exponential backoff."""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from typing import Final

from dqliteclient.exceptions import ClusterError, ClusterPolicyError, DqliteConnectionError

__all__ = ["retry_with_backoff"]

# OS-entropy randomness for the jitter draw, mirroring the
# ``_cluster_random`` discipline at ``cluster.py:159-165``. The
# module-global ``random._inst`` (used by bare ``random.uniform``)
# is inherited byte-for-byte across ``os.fork()`` — CPython's
# ``Lib/random.py`` does not register an ``os.register_at_fork``
# callback for it. Forked workers (gunicorn/uwsgi/multiprocessing
# fork start method) therefore draw IDENTICAL jitter sequences from
# their inherited state, defeating the stagger that jitter is meant
# to provide. ``SystemRandom`` reads ``/dev/urandom`` per draw
# (Linux ``getrandom(2)`` is fork-aware) so siblings diverge; it
# also ignores ``random.seed()`` so a test suite that seeds the
# global PRNG for determinism cannot accidentally pin every retry
# to the same jitter draw.
_retry_random: Final[random.Random] = random.SystemRandom()

# Default retry set: transport- and cluster-level errors only.
# Deterministic server / client errors (``OperationalError``,
# ``DataError``, ``InterfaceError``) are NOT retried — retrying a
# constraint violation or a type mismatch would burn the full
# exponential-backoff window before surfacing the true cause. A
# caller that actually wants broader catches must opt in by passing
# an explicit ``retryable_exceptions`` tuple.
#
# ``OSError`` subsumes ``TimeoutError``, ``BrokenPipeError``,
# ``ConnectionError``, and ``ConnectionResetError``, so a single
# ``OSError`` entry covers every stdlib transport-error shape
# (mirrors the classification in
# ``sqlalchemy-dqlite/src/sqlalchemydqlite/base.py``'s
# ``is_disconnect``).
_DEFAULT_RETRYABLE: Final[tuple[type[BaseException], ...]] = (
    OSError,
    DqliteConnectionError,
    ClusterError,
)

# ``ClusterPolicyError`` is a deterministic subclass of
# ``ClusterError``: the redirect-policy gate rejected an address, and
# retrying the same RPC against the same configured policy will land
# on the same rejection. Excluding it by default short-circuits the
# misuse where a caller wraps a policy-bounded RPC and burns the full
# exponential-backoff window before surfacing the policy hit. The
# in-tree ``ClusterClient.connect`` path passes the same exclusion
# explicitly (``cluster.py``); external callers using the default get
# the same behaviour without having to know about the subclass.
_DEFAULT_EXCLUDED: Final[tuple[type[BaseException], ...]] = (ClusterPolicyError,)


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    jitter: float = 0.1,
    max_elapsed_seconds: float | None = None,
    retryable_exceptions: tuple[type[BaseException], ...] = _DEFAULT_RETRYABLE,
    excluded_exceptions: tuple[type[BaseException], ...] = _DEFAULT_EXCLUDED,
) -> T:
    """Retry an async function with exponential backoff.

    Safety note for SQL callables:
        Retrying a callable that issues SQL against dqlite is UNSAFE for
        non-idempotent statements. The cluster commits writes through
        Raft replication; a transport-level failure (TCP reset, leader
        flip, deadline exceeded) can fire AFTER the server has already
        applied and Raft-replicated the change but BEFORE the client
        receives the success response. Re-running the statement on
        retry produces duplicate rows / double-applied updates with no
        idempotence guarantees from the protocol.

        The default ``retryable_exceptions`` tuple deliberately omits
        ``OperationalError`` so the most common SQL-level failures do
        not retry. Callers who broaden the tuple — or who wrap a SQL
        callable in this helper at all — must restrict themselves to
        idempotent shapes (UPSERT, INSERT OR IGNORE, plain SELECT,
        application-level idempotency tokens) or accept at-least-once
        application semantics.

    Args:
        func: Async function to retry. See the safety note above before
            wrapping a SQL-mutating callable.
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        jitter: Random jitter factor in ``[0, 1)`` — e.g. ``0.5``
            means each delay is randomly scaled by ``[0.5, 1.5]``.
            ``1.0`` is rejected: ``1 + uniform(-1, 1)`` can legally
            draw 0, zeroing the backoff for that iteration and
            defeating the exponential-backoff contract. Cap at
            ``0.99`` for the strongest randomisation.
        max_elapsed_seconds: Optional total wall-clock cap. ``None``
            (default) means "only ``max_attempts`` governs termination."
            Set to a positive finite number to abort the retry loop
            once the cumulative elapsed time crosses the budget,
            re-raising the last observed exception. Complements
            ``max_attempts`` for scenarios where a single attempt has
            its own long deadline (``ClusterClient.find_leader`` with
            N slow peers, for example): without this cap the worst-case
            wall clock is ``max_attempts * (per-attempt deadline +
            max_delay)``, easily exceeding a caller's outer deadline.
        retryable_exceptions: Exception types to retry on. The default
            (``OSError``, ``DqliteConnectionError``, ``ClusterError``)
            covers transport- and cluster-level failures only —
            deterministic server/client errors (``OperationalError``,
            ``DataError``, ``InterfaceError``) are NOT retried by
            default. Callers that want broader catches should pass
            their own tuple, but see the safety note above on the
            at-least-once consequences of broadening this for SQL
            callables.
        excluded_exceptions: Subclasses of ``retryable_exceptions`` that
            must NOT be retried — useful when a deterministic,
            non-recoverable error (e.g. a policy rejection) is a subtype
            of an otherwise-retryable family. Matched before the
            retryable check so the exception re-raises immediately.
            Default: ``(ClusterPolicyError,)`` — a redirect-policy
            rejection is deterministic against the configured policy
            and retrying would burn the full backoff window for the
            same outcome. Pass an empty tuple to opt back into the
            previous "retry every matching subclass" behaviour.

    Returns:
        Result of the function

    Raises:
        The last exception if all attempts fail, or immediately
        for non-retryable exceptions

    Comparison to ``ClusterClient.connect``:
        The in-tree connect path invokes this helper with TIGHTER
        defaults than the public signature for go-dqlite parity:
        ``max_attempts=3`` (vs. 5 here) and ``max_delay=1.0`` (vs.
        10.0 here), mirroring go-dqlite's ``BackoffCap = 1 * time.Second``.
        External callers wrapping their own cluster-layer RPCs and
        wanting that same shape should pass ``max_attempts=3``,
        ``max_delay=1.0`` explicitly. The public defaults are looser
        because non-cluster callers (admin tooling, third-party
        wrappers) reasonably tolerate a longer worst-case wall-clock
        in exchange for more attempts.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError(f"max_attempts must be an int, got {type(max_attempts).__name__}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    # Validate the float knobs: a negative or non-finite base_delay /
    # max_delay would later be handed to ``asyncio.sleep`` which rejects
    # negative values with its own ValueError — surface the misuse at
    # the retry helper's entry point instead, with a clearer message.
    # ``jitter > 1`` is the same class of bug: ``1 + uniform(-j, j)``
    # with ``j > 1`` can go negative and produce a negative sleep.
    for name, value in (("base_delay", base_delay), ("max_delay", max_delay)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number, got {value}")
    if isinstance(jitter, bool) or not isinstance(jitter, (int, float)):
        raise TypeError(f"jitter must be a number, got {type(jitter).__name__}")
    if not math.isfinite(jitter) or not (0 <= jitter < 1):
        # Half-open interval ``[0, 1)``: at ``jitter=1.0`` exactly,
        # ``1 + uniform(-1, 1)`` can legally draw 0 (the lower bound
        # is a legal return of ``random.uniform(a, b)``), which zeros
        # the backoff for that iteration and defeats the documented
        # exponential-backoff contract. Reject at the validator
        # rather than admitting the degenerate value silently.
        raise ValueError(
            f"jitter must be in [0, 1) — values at or above 1 allow the "
            f"uniform(-jitter, jitter) draw to zero the backoff, defeating "
            f"the exponential-backoff contract. Got {jitter}."
        )
    if max_elapsed_seconds is not None:
        if isinstance(max_elapsed_seconds, bool) or not isinstance(
            max_elapsed_seconds, (int, float)
        ):
            raise TypeError(
                f"max_elapsed_seconds must be a number or None, "
                f"got {type(max_elapsed_seconds).__name__}"
            )
        if not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0:
            raise ValueError(
                f"max_elapsed_seconds must be a positive finite number, got {max_elapsed_seconds}"
            )

    loop = asyncio.get_running_loop()
    deadline = None if max_elapsed_seconds is None else loop.time() + max_elapsed_seconds

    last_error: BaseException | None = None
    history: list[BaseException] = []

    for attempt in range(max_attempts):
        # Deadline re-check BEFORE the next func() call: a previous
        # backoff sleep may have woken close to the deadline. Without
        # this check the next func() runs entirely past the deadline,
        # producing wall-clock overshoots that are visible to outer
        # asyncio.wait_for callers as missed timeouts.
        if attempt > 0 and deadline is not None and loop.time() >= deadline:
            break
        try:
            return await func()
        except excluded_exceptions:
            # Deterministic-failure subclass of a retryable family —
            # retrying would just reproduce it. Re-raise immediately.
            raise
        except BaseExceptionGroup as bg:
            # PEP 654: ``func`` that wraps work in ``asyncio.TaskGroup``
            # (or any structured-concurrency primitive that re-raises
            # children as a group) escapes the leaf ``retryable_exceptions``
            # arm below because ``BaseExceptionGroup`` is not a subclass
            # of the leaf classes. Unwrap with ``split`` so a group whose
            # every leaf is retryable participates in the retry loop —
            # mirrors the recursive-unwrap discipline at
            # ``_run_protocol``'s cancel-detection arm.
            #
            # Fail-fast on a mixed group containing any excluded leaf
            # so a deterministic failure does not get masked as
            # transient.
            excluded_matched, _ = bg.split(excluded_exceptions)
            if excluded_matched is not None:
                raise
            # Retryable group: every leaf must match the retryable set
            # for the retry to be safe. ``unmatched is not None`` means
            # at least one leaf is neither retryable nor excluded — a
            # genuinely unexpected exception class slipping through;
            # do not silently retry on it.
            matched, unmatched = bg.split(retryable_exceptions)
            if unmatched is not None or matched is None:
                raise
            last_error = matched
            history.append(matched)
        except retryable_exceptions as e:
            last_error = e
            history.append(e)

        # Shared retry-loop continuation reached from both the
        # ``BaseExceptionGroup`` arm and the leaf ``retryable_exceptions``
        # arm. Either set ``last_error`` and ``history`` before falling
        # through here.
        if attempt == max_attempts - 1:
            break
        # Budget check BEFORE scheduling the sleep: if we're out
        # of time, re-raise the last error now rather than burn
        # another backoff.
        if deadline is not None and loop.time() >= deadline:
            break

        # Calculate delay with exponential backoff
        delay = min(base_delay * (2**attempt), max_delay)

        # Add jitter, then re-clamp so max_delay stays a hard ceiling.
        # ``_retry_random`` (SystemRandom) is used over ``random.uniform``
        # to keep forked workers' jitter draws independent — see the
        # module-level comment on ``_retry_random``.
        if jitter > 0:
            delay = min(delay * (1 + _retry_random.uniform(-jitter, jitter)), max_delay)

        # Clamp the sleep so it never straddles the deadline.
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - loop.time()))

        await asyncio.sleep(delay)

    # last_error is non-None here: the break on the final attempt only
    # runs after ``last_error = e`` executes, and max_attempts < 1 is
    # rejected above. The assert pins the loop invariant for mypy
    # without runtime cost on production paths (a stripped-by-O assert
    # would still leave the invariant intact at this point because the
    # break path always sets last_error first).
    assert last_error is not None
    if len(history) > 1:
        # Chain prior-attempt failures so a forensic walker can see
        # the full timeline rather than only the last error. Mirrors
        # the discipline ``_find_leader_impl`` and
        # ``ConnectionPool.initialize`` apply for per-node aggregates.
        # ``_bounded_group`` caps children so the chain stays
        # picklable for cross-process error capture. Local import to
        # avoid the retry <-> cluster import cycle at module load.
        from dqliteclient.cluster import _bounded_group

        raise last_error from _bounded_group(
            f"retry exhausted after {len(history)} attempts", history
        )
    raise last_error
