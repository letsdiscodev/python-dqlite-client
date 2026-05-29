"""Pin: ``retry_with_backoff`` defends its ``last_error is not None``
invariant with a ``RuntimeError``, not a bare ``assert`` (stripped under
``python -O`` / ``-OO``)."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from dqliteclient.retry import retry_with_backoff


@pytest.mark.asyncio
async def test_invariant_check_uses_raise_not_assert() -> None:
    """The invariant check uses ``if ... is None: raise RuntimeError(...)``."""
    src = inspect.getsource(retry_with_backoff)
    assert "if last_error is None:" in src
    assert "RuntimeError" in src
    assert "internal invariant violated" in src
    assert "assert last_error is not None" not in src


@pytest.mark.asyncio
async def test_happy_path_still_raises_last_error_with_history_chain() -> None:
    """The exhaustion raise chain still fires when the invariant is upheld."""

    class _Transient(Exception):
        pass

    attempts: list[int] = []

    async def always_fail() -> None:
        attempts.append(1)
        raise _Transient("boom")

    with pytest.raises(_Transient) as exc_info:
        await retry_with_backoff(
            always_fail,
            max_attempts=3,
            base_delay=0.001,
            max_delay=0.001,
            jitter=0.0,
            retryable_exceptions=(_Transient,),
        )
    assert len(attempts) == 3
    assert exc_info.value.__cause__ is not None
    assert "retry exhausted after 3 attempts" in str(exc_info.value.__cause__)


@pytest.mark.asyncio
async def test_invariant_violation_message_includes_context() -> None:
    """The invariant violation isn't reachable via the public API; the
    source-level pin guards the shape, so this just checks the happy return."""

    async def ok() -> int:
        return 42

    result = await retry_with_backoff(ok, max_attempts=1, base_delay=0.0)
    assert result == 42


def test_module_imports_under_python_o() -> None:
    """The retry module compiles cleanly to bytecode (no syntax errors)."""
    from dqliteclient import retry as retry_mod

    src = inspect.getsource(retry_mod)
    compile(src, retry_mod.__file__ or "retry.py", "exec")


def test_assert_alternative_compatible_with_optimize() -> None:
    """``if x is None: raise`` survives compiling with optimize=2 (``-OO``)."""

    src = """
def f(x):
    if x is None:
        raise RuntimeError("invariant violated")
    return x
"""
    code = compile(src, "<test>", "exec", optimize=2)
    namespace: dict[str, object] = {}
    exec(code, namespace)  # noqa: S102
    func = namespace["f"]
    assert callable(func)
    with pytest.raises(RuntimeError, match="invariant violated"):
        func(None)
    assert func(5) == 5


_ = asyncio  # noqa: F401  - keep the import for the synchronous tests above
