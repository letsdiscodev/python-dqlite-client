"""Cluster shuffle uses ``random.SystemRandom`` so ``random.seed()`` cannot defeat it."""

from __future__ import annotations

import random

from dqliteclient.cluster import _cluster_random


def test_cluster_random_is_system_random() -> None:
    assert isinstance(_cluster_random, random.SystemRandom)


def test_cluster_shuffle_unaffected_by_seed() -> None:
    """Identical global seeds must still yield different cluster shuffles."""
    nodes_first: list[int] = list(range(100))
    nodes_second: list[int] = list(range(100))

    random.seed(42)
    _cluster_random.shuffle(nodes_first)

    random.seed(42)
    _cluster_random.shuffle(nodes_second)

    assert nodes_first != nodes_second, (
        "cluster shuffle must use OS entropy, not the seeded module PRNG"
    )


def test_cluster_shuffle_does_not_consume_module_random_state() -> None:
    """A cluster shuffle between two module ``random.random()`` draws must not affect them."""
    random.seed(123)
    _ = random.random()
    expected_next = random.random()

    random.seed(123)
    _ = random.random()
    _cluster_random.shuffle(list(range(100)))
    actual_next = random.random()

    assert actual_next == expected_next, (
        "cluster shuffle must not consume bytes from the global module-level Random instance"
    )
