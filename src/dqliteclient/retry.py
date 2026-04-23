"""Retry utilities with exponential backoff."""

import asyncio
import random
from collections.abc import Awaitable, Callable

from dqliteclient.exceptions import ClusterError, DqliteConnectionError

# Default retry set: transport- and cluster-level errors only.
# Deterministic server / client errors (``OperationalError``,
# ``DataError``, ``InterfaceError``) are NOT retried — retrying a
# constraint violation or a type mismatch would burn the full
# exponential-backoff window before surfacing the true cause. A
# caller that actually wants broader catches must opt in by passing
# an explicit ``retryable_exceptions`` tuple.
#
# ``OSError`` subsumes ``TimeoutError``, ``BrokenPipeError``,
# ``ConnectionError``, and ``ConnectionResetError`` since Python
# 3.10, so a single ``OSError`` entry covers every stdlib
# transport-error shape (mirrors the classification in
# ``sqlalchemy-dqlite/src/sqlalchemydqlite/base.py``'s
# ``is_disconnect``). The package requires Python 3.13+.
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
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

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
