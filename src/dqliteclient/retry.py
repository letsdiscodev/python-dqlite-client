"""Retry utilities with exponential backoff."""

import asyncio
import random
from collections.abc import Awaitable, Callable


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    jitter: float = 0.1,
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        jitter: Random jitter factor (0-1)

    Returns:
        Result of the function

    Raises:
        The last exception if all attempts fail
    """
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_error = e

            if attempt == max_attempts - 1:
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2**attempt), max_delay)

            # Add jitter
            if jitter > 0:
                delay = delay * (1 + random.uniform(-jitter, jitter))

            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
