"""Integration test fixtures for dqlite-client.

These tests require a running dqlite cluster.
Start the test cluster with:
    cd ../python-dqlite-dev/cluster && docker compose up -d

Note: the ``pytest_plugins = ["dqlitetestlib.fixtures"]`` declaration
that would normally live next to integration-only fixtures is in the
top-level ``tests/conftest.py`` instead. Pytest 8+ rejects
``pytest_plugins`` in non-top-level conftests.
"""

import os

import pytest

# Single bootstrap address. Default matches the python-dqlite-dev
# cluster (canonical dqlite port 9001 on host networking).
DQLITE_TEST_CLUSTER = os.environ.get("DQLITE_TEST_CLUSTER", "localhost:9001")
# Full node list. Default matches the python-dqlite-dev cluster's
# three nodes (9001-9003 on host networking). Override via
# ``DQLITE_TEST_CLUSTER_NODES`` (comma-separated).
DQLITE_TEST_CLUSTER_NODES = os.environ.get(
    "DQLITE_TEST_CLUSTER_NODES",
    "localhost:9001,localhost:9002,localhost:9003",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as requiring dqlite cluster")


@pytest.fixture
def cluster_address() -> str:
    """Get the test cluster address."""
    return DQLITE_TEST_CLUSTER


@pytest.fixture
def cluster_addresses() -> list[str]:
    """Get all test cluster addresses."""
    return ["localhost:9001", "localhost:9002", "localhost:9003"]


@pytest.fixture
def cluster_node_addresses() -> list[str]:
    """All host-mapped cluster node addresses, configurable via the
    ``DQLITE_TEST_CLUSTER_NODES`` env var. Use this for multi-node
    tests that need every node reachable (the older
    ``cluster_addresses`` fixture is hardcoded and kept for
    backwards-compatibility with the shared
    ``dqlite-test-cluster/pytest_fixtures.py`` shape)."""
    return [a.strip() for a in DQLITE_TEST_CLUSTER_NODES.split(",") if a.strip()]
