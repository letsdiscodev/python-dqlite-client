"""Retry utilities with exponential backoff."""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

from dqliteclient.exceptions import ClusterError, DqliteConnectionError

__all__ = ["retry_with_backoff"]

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
_DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    OSError,
    DqliteConnectionError,
    ClusterError,
)


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple[type[BaseException], ...] = _DEFAULT_RETRYABLE,
    excluded_exceptions: tuple[type[BaseException], ...] = (),
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        jitter: Random jitter factor (0-1)
        retryable_exceptions: Exception types to retry on. The default
            (``OSError``, ``DqliteConnectionError``, ``ClusterError``)
            covers transport- and cluster-level failures only —
            deterministic server/client errors (``OperationalError``,
            ``DataError``, ``InterfaceError``) are NOT retried by
            default. Callers that want broader catches should pass
            their own tuple.
        excluded_exceptions: Subclasses of ``retryable_exceptions`` that
            must NOT be retried — useful when a deterministic,
            non-recoverable error (e.g. a policy rejection) is a subtype
            of an otherwise-retryable family. Matched before the
            retryable check so the exception re-raises immediately.

    Returns:
        Result of the function

    Raises:
        The last exception if all attempts fail, or immediately
        for non-retryable exceptions
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
    if not math.isfinite(jitter) or not (0 <= jitter <= 1):
        raise ValueError(f"jitter must be in [0, 1], got {jitter}")

    last_error: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except excluded_exceptions:
            # Deterministic-failure subclass of a retryable family —
            # retrying would just reproduce it. Re-raise immediately.
            raise
        except retryable_exceptions as e:
            last_error = e

            if attempt == max_attempts - 1:
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2**attempt), max_delay)

            # Add jitter, then re-clamp so max_delay stays a hard ceiling.
            if jitter > 0:
                delay = min(delay * (1 + random.uniform(-jitter, jitter)), max_delay)

            await asyncio.sleep(delay)

    # last_error is non-None here: the break on the final attempt only
    # runs after ``last_error = e`` executes, and max_attempts < 1 is
    # rejected above. mypy can't see the loop invariant.
    raise last_error  # type: ignore[misc]
