"""Pin: ``ConnectionPool._release`` drains ``conn._pending_drain``
at every exit shape (the three early-return arms + the finally),
via the centralised ``_drain_pending_under_shield`` helper.

The finally-branch's load-bearing comment documented the necessity
of an explicit ``_pending_drain`` snapshot + shield-await BEFORE
``_pool_released=True`` flips and short-circuits the in-``close()``
re-snapshot loop. The three early-return arms were silent
free-riders on ``_close_impl``'s internals — a future refactor of
the re-snapshot loop would silently regress drain hygiene at all
three sites without anything failing loudly. Factor the drain into
a shared helper and call it at every exit so the discipline is
symmetric and self-documenting.
"""

from __future__ import annotations

import inspect

from dqliteclient import pool as pool_mod


def test_release_calls_drain_helper_at_every_early_return_arm() -> None:
    """Every early-return ``_close_best_effort`` call inside ``_release``
    is followed by ``_drain_pending_under_shield(conn)`` so the
    centralised drain discipline fires at every early-return exit
    shape (the queuefull path falls through to the finally arm,
    which also calls the helper — covered by the separate finally
    pin below).
    """
    src = inspect.getsource(pool_mod.ConnectionPool._release)
    lines = [line.rstrip() for line in src.splitlines()]
    # The early-return arms use the labels "release-closed",
    # "release-reset-rolled-back", "release-post-reset-closed". The
    # "release-queuefull" call does NOT return — it falls through to
    # the finally where the drain helper also fires.
    early_return_labels = (
        "release-closed",
        "release-reset-rolled-back",
        "release-post-reset-closed",
    )
    for label in early_return_labels:
        # Find the close_best_effort call line for this label.
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
    """The finally arm also calls the helper (rather than inlining
    the pending-drain snapshot). Verifies the refactor is complete —
    not just additive at the early-return sites.
    """
    src = inspect.getsource(pool_mod.ConnectionPool._release)
    # The finally arm should no longer contain the inline snapshot.
    assert "pending = getattr(conn, " not in src, (
        "_release's finally arm must use _drain_pending_under_shield "
        "instead of inlining the pending snapshot — centralisation "
        "is the load-bearing change"
    )
    # The helper IS called in the finally — appears at least once
    # alongside the early-return arms.
    assert src.count("await self._drain_pending_under_shield(conn)") >= 4, (
        "expected the helper call in all four exit shapes (three early-returns + the finally)"
    )
