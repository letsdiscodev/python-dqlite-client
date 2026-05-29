"""Pin: the late-winner helper's RuntimeError is absorbed in ``acquire``'s except-arm
cleanup so a racing loop-close does not supplant the user's original cancel."""

from __future__ import annotations

import inspect

from dqliteclient import pool as pool_mod


def test_late_winner_helper_calls_are_runtime_error_protected() -> None:
    src = inspect.getsource(pool_mod.ConnectionPool.acquire)
    lines = [line for line in src.splitlines() if line.strip()]
    helper_call_indices = [
        i for i, line in enumerate(lines) if "await self._put_back_or_release_late_winner(" in line
    ]
    assert helper_call_indices, "expected at least one late-winner helper call site"
    for idx in helper_call_indices:
        prev = lines[idx - 1].strip()
        scan = idx - 1
        while prev.startswith("#") and scan > 0:
            scan -= 1
            prev = lines[scan].strip()
        assert prev == "try:", (
            f"late-winner helper call at line {idx} of acquire() must be "
            f"wrapped in try/except RuntimeError so a helper RuntimeError "
            f"does not supplant the user's original cancel; previous "
            f"non-comment line is {prev!r}"
        )
    body = "\n".join(lines)
    assert body.count("except RuntimeError:") >= len(helper_call_indices), (
        "every late-winner helper call site must have a paired "
        "`except RuntimeError:` arm; counts mismatch"
    )
