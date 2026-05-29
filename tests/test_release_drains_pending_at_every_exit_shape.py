"""Pin: ``ConnectionPool._release`` drains via the shared
``_drain_pending_under_shield`` helper at all four exit shapes."""

from __future__ import annotations

import inspect

from dqliteclient import pool as pool_mod


def test_release_calls_drain_helper_at_every_early_return_arm() -> None:
    """Each early-return arm is followed by a drain-helper call."""
    src = inspect.getsource(pool_mod.ConnectionPool._release)
    lines = [line.rstrip() for line in src.splitlines()]
    # "release-queuefull" is excluded: it falls through to the finally, not a return.
    early_return_labels = (
        "release-closed",
        "release-reset-rolled-back",
        "release-post-reset-closed",
    )
    for label in early_return_labels:
        site_idx = next(
            (i for i, line in enumerate(lines) if f'"{label}"' in line),
            None,
        )
        assert site_idx is not None, f"expected to find _close_best_effort call site for {label!r}"
        # Skip blank / comment lines.
        scan = site_idx + 1
        while scan < len(lines) and (
            not lines[scan].strip() or lines[scan].strip().startswith("#")
        ):
            scan += 1
        next_code = lines[scan].strip() if scan < len(lines) else ""
        assert "await self._drain_pending_under_shield(conn)" in next_code, (
            f"{label} early-return arm must call "
            f"`await self._drain_pending_under_shield(conn)` immediately "
            f"after _close_best_effort; found {next_code!r}"
        )


def test_drain_pending_under_shield_helper_exists() -> None:
    """Pin the helper's existence so the test above stays meaningful."""
    assert hasattr(pool_mod.ConnectionPool, "_drain_pending_under_shield"), (
        "ConnectionPool must expose _drain_pending_under_shield so all "
        "four _release exit shapes can share the drain discipline"
    )


def test_release_finally_uses_drain_helper() -> None:
    """The finally arm uses the helper, not an inline pending snapshot."""
    src = inspect.getsource(pool_mod.ConnectionPool._release)
    assert "pending = getattr(conn, " not in src, (
        "_release's finally arm must use _drain_pending_under_shield "
        "instead of inlining the pending snapshot — centralisation "
        "is the load-bearing change"
    )
    assert src.count("await self._drain_pending_under_shield(conn)") >= 4, (
        "expected the helper call in all four exit shapes (three early-returns + the finally)"
    )
