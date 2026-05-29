"""Retry utilities with exponential backoff."""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from typing import Final

from dqliteclient.exceptions import ClusterError, ClusterPolicyError, DqliteConnectionError

__all__ = ["retry_with_backoff"]

# SystemRandom (not bare random.uniform) so forked workers draw independent jitter:
# random._inst is inherited byte-for-byte across os.fork() and would otherwise stagger-defeat.
_retry_random: Final[random.Random] = random.SystemRandom()

# Transport/cluster errors only; deterministic server/client errors (OperationalError etc.)
# are NOT retried by default. OSError subsumes TimeoutError/BrokenPipeError/ConnectionError.
_DEFAULT_RETRYABLE: Final[tuple[type[BaseException], ...]] = (
    OSError,
    DqliteConnectionError,
    ClusterError,
)

# ClusterPolicyError is a deterministic ClusterError subclass (policy gate rejected an
# address); retrying the same RPC against the same policy reproduces it, so exclude by default.
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

    UNSAFE for non-idempotent SQL: a transport failure can fire after the write is
    Raft-committed but before the client sees success, so retry duplicates the write.
    The default retryable set omits OperationalError to avoid retrying SQL-level failures.

    jitter must be in [0, 1): at 1.0, ``1 + uniform(-1, 1)`` can draw 0 and zero the backoff.
    max_elapsed_seconds is an optional wall-clock cap complementing max_attempts.
    excluded_exceptions are non-retryable subclasses of retryable_exceptions, matched first.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError(f"max_attempts must be an int, got {type(max_attempts).__name__}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    # Surface negative/non-finite delays here rather than later inside asyncio.sleep.
    for name, value in (("base_delay", base_delay), ("max_delay", max_delay)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number, got {value}")
    if isinstance(jitter, bool) or not isinstance(jitter, (int, float)):
        raise TypeError(f"jitter must be a number, got {type(jitter).__name__}")
    if not math.isfinite(jitter) or not (0 <= jitter < 1):
        # At jitter=1.0, ``1 + uniform(-1, 1)`` can draw 0 and zero the backoff.
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
        # Deadline re-check before func(): a prior backoff sleep may have woken near it.
        if attempt > 0 and deadline is not None and loop.time() >= deadline:
            break
        try:
            return await func()
        except excluded_exceptions:
            raise
        except BaseExceptionGroup as bg:
            # PEP 654: unwrap groups (e.g. from asyncio.TaskGroup) since the group itself
            # is not a subclass of the leaf retryable types. Fail fast if any leaf is excluded.
            excluded_matched, _ = bg.split(excluded_exceptions)
            if excluded_matched is not None:
                raise
            # Retry only if every leaf is retryable; unmatched means an unexpected leaf.
            matched, unmatched = bg.split(retryable_exceptions)
            if unmatched is not None or matched is None:
                raise
            last_error = matched
            history.append(matched)
        except retryable_exceptions as e:
            last_error = e
            history.append(e)

        if attempt == max_attempts - 1:
            break
        if deadline is not None and loop.time() >= deadline:
            break

        delay = min(base_delay * (2**attempt), max_delay)

        # Add jitter, then re-clamp so max_delay stays a hard ceiling.
        if jitter > 0:
            delay = min(delay * (1 + _retry_random.uniform(-jitter, jitter)), max_delay)

        # Clamp the sleep so it never straddles the deadline.
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - loop.time()))

        await asyncio.sleep(delay)

    # last_error is non-None here; raise (not assert) so the invariant survives python -O.
    if last_error is None:
        raise RuntimeError(
            f"retry_with_backoff: internal invariant violated — exited "
            f"retry loop with last_error=None (max_attempts={max_attempts}, "
            f"history_len={len(history)}). This is a bug in the retry helper."
        )
    if len(history) > 1:
        # Chain prior failures so the full timeline is visible, not just the last error.
        # _bounded_group caps children to stay picklable. Local import breaks an import cycle.
        from dqliteclient.cluster import _bounded_group

        raise last_error from _bounded_group(
            f"retry exhausted after {len(history)} attempts", history
        )
    raise last_error
